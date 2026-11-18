"""PathSnap shared core: geometry, inference-only candidate/feature builder, and training-time labeller.

This module is imported by the training scripts (feature.py), the HMM decoder, and the FastAPI backend,
so that training and serving compute exactly the same features.

Split of responsibilities (leakage control):
  * build_candidates(track, nodes, arcs)            -> uses GPS + road network ONLY (safe for inference)
  * label_points(ctx, route)                        -> uses the reference .route (training / server-side scoring only)
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

K_MAX, R_CAND, STEP = 15, 150.0, 20.0        # candidate rule: nearest K arcs within R metres (STEP = index spacing)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")                 # all CSV / JSON results
MODELS = os.path.join(REPO, "models")             # saved model + decoder config
FIGS = os.path.join(REPO, "figures")              # plots
MAPS = os.path.join(REPO, "maps")                 # Folium HTML maps
FEATS = ["d", "rank", "d_rel", "t", "L", "cos_h", "side_d", "dend", "d_prev", "d_next", "twin", "indeg_u", "outdeg_v",
         "n_cand", "speed", "dt_prev", "step_prev", "turn_cos", "hvalid"]


def find_student_root():
    """Folder that contains train/ and student_test/ (the organisers' 'Student folder')."""
    for r, d, _ in os.walk(os.path.join(REPO, "Problem-Statement")):
        if "train" in d and "student_test" in d:
            return r
    raise FileNotFoundError("Could not find the folder containing train/ and student_test/ under Problem-Statement/")


def data_path(name):
    return os.path.join(DATA, name)


# ----------------------------------------------------------------------------------------------- file reading
def read_table(path):
    return pd.read_csv(path, sep=r"\s+", header=None).values


def load_trip_inputs(base):
    """Inputs allowed for prediction: .track (lon, lat, ts), .nodes (lon, lat), .arcs (from, to). No .route."""
    return (read_table(base + ".track").astype(float), read_table(base + ".nodes").astype(float),
            read_table(base + ".arcs").astype(np.int64))


def load_route(base):
    """Reference route (ordered arc ids). Only for training labels and server-side scoring."""
    return read_table(base + ".route")[:, 0].astype(np.int64)


def trip_base(folder, tid):
    return os.path.join(folder, tid, tid)


# ----------------------------------------------------------------------------------------------- geometry
def project(xy, lon0, lat0):
    """Local equirectangular projection (lon, lat) -> metres, centred on the trip mean. Accurate at ~10 km scale."""
    return np.c_[(xy[:, 0] - lon0) * 111320 * np.cos(np.deg2rad(lat0)), (xy[:, 1] - lat0) * 110540.0]


def unproject(P, lon0, lat0):
    return np.c_[P[:, 0] / (111320 * np.cos(np.deg2rad(lat0))) + lon0, P[:, 1] / 110540.0 + lat0]


def seg_dist(P, A, B):
    """(n_points, n_segments) point-to-segment distance in metres."""
    AB = B - A; L2 = np.maximum((AB ** 2).sum(1), 1e-9); out = np.empty((len(P), len(A)), np.float32)
    for s in range(0, len(P), 2000):
        p = P[s:s + 2000, None, :]
        t = np.clip(((p - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
        out[s:s + 2000] = np.sqrt(((p - (A[None] + t[..., None] * AB[None])) ** 2).sum(2))
    return out


def seg_geom(p, A, B):
    """One point vs many segments -> distance, fraction t along segment (0..1), side (+1 left / -1 right)."""
    AB = B - A; L2 = np.maximum((AB ** 2).sum(1), 1e-9); v = p - A
    t = np.clip((v * AB).sum(1) / L2, 0, 1)
    d = np.linalg.norm(p - (A + t[:, None] * AB), axis=1)
    side = np.where(AB[:, 0] * v[:, 1] - AB[:, 1] * v[:, 0] >= 0, 1.0, -1.0)
    return d, t, side


def align(D):
    """Monotone DP: GPS point i -> route position j(i), j non-decreasing, minimising the summed distance."""
    n, m = D.shape; idx = np.arange(m); cost = D[0].astype(np.float64); back = np.empty((n, m), np.int32); back[0] = idx
    for i in range(1, n):
        macc = np.minimum.accumulate(cost)
        back[i] = np.maximum.accumulate(np.where(cost == macc, idx, 0)); cost = D[i] + macc
    j = int(cost.argmin()); path = np.empty(n, np.int32)
    for i in range(n - 1, -1, -1):
        path[i] = j; j = back[i][j]
    return path


# ----------------------------------------------------------------------------------------------- inference-only features
def build_candidates(track, nodes, arcs):
    """Candidate generation + 19 features per (GPS point, candidate arc) row. Uses NO reference route.

    Returns X (DataFrame: gi, arc, FEATS...) and ctx (dict of projected geometry reused by the HMM / labeller).
    """
    t, nd, ar = track, nodes, arcs
    n, nn = len(t), len(nd)
    lon0, lat0 = t[:, 0].mean(), t[:, 1].mean()
    P, N = project(t[:, :2], lon0, lat0), project(nd, lon0, lat0)
    A, B = N[ar[:, 0]], N[ar[:, 1]]; AB = B - A; L = np.linalg.norm(AB, axis=1)
    key = ar[:, 0] * nn + ar[:, 1]                                   # arcs with same (from,to) are equivalent
    has_twin = np.isin(ar[:, 1] * nn + ar[:, 0], key).astype(np.float32)
    indeg_u = np.bincount(ar[:, 1], minlength=nn)[ar[:, 0]].astype(np.float32)   # in-degree of arc start node
    outdeg_v = np.bincount(ar[:, 0], minlength=nn)[ar[:, 1]].astype(np.float32)  # out-degree of arc end node

    # GPS-only context features
    ts = t[:, 2]; dt_prev = np.r_[1.0, np.diff(ts)]; step_prev = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)]
    speed = step_prev / np.maximum(dt_prev, 1e-9)
    ia, ib = np.clip(np.arange(n) - 2, 0, n - 1), np.clip(np.arange(n) + 2, 0, n - 1)
    hv = P[ib] - P[ia]; hn = np.linalg.norm(hv, axis=1); hvalid = hn > 5.0          # heading from a +/-2 point window
    hu = np.where(hvalid[:, None], hv / np.maximum(hn, 1e-9)[:, None], 0.0)
    turn_cos = np.r_[1.0, (hu[1:] * hu[:-1]).sum(1)]
    ip, inx = np.clip(np.arange(n) - 1, 0, n - 1), np.clip(np.arange(n) + 1, 0, n - 1)

    # spatial index: sample points every ~STEP metres along every arc
    ns = np.maximum(2, np.ceil(L / STEP).astype(int) + 1)
    arc_idx = np.repeat(np.arange(len(ar)), ns); kk = np.arange(ns.sum()) - np.repeat(np.cumsum(ns) - ns, ns)
    tree = cKDTree(A[arc_idx] + (kk / np.repeat(ns - 1, ns))[:, None] * AB[arc_idx])

    cols = {k: [] for k in ["gi", "arc", "d", "rank", "d_rel", "t", "L", "cos_h", "side_d", "dend", "d_prev", "d_next",
                            "twin", "indeg_u", "outdeg_v", "n_cand"]}
    for s in range(0, n, 2000):
        nbs = tree.query_ball_point(P[s:s + 2000], r=R_CAND + STEP / 2 + 1, workers=-1)
        for off, nb in enumerate(nbs):
            i = s + off
            if len(nb) == 0: continue
            c = np.unique(arc_idx[np.asarray(nb, dtype=int)])
            d, tt, sd = seg_geom(P[i], A[c], B[c]); m = d <= R_CAND
            if not m.any(): continue
            c, d, tt, sd = c[m], d[m], tt[m], sd[m]
            o = np.argsort(d, kind="stable")[:K_MAX]; c, d, tt, sd = c[o], d[o], tt[o], sd[o]; k = len(c)
            av = AB[c] / np.maximum(L[c], 1e-9)[:, None]
            cols["gi"].append(np.full(k, i)); cols["arc"].append(c); cols["d"].append(d); cols["rank"].append(np.arange(k))
            cols["d_rel"].append(d - d[0]); cols["t"].append(tt); cols["L"].append(L[c])
            cols["cos_h"].append(av @ hu[i] if hvalid[i] else np.zeros(k)); cols["side_d"].append(sd * d)
            cols["dend"].append(np.minimum(tt * L[c], (1 - tt) * L[c]))
            cols["d_prev"].append(seg_geom(P[ip[i]], A[c], B[c])[0]); cols["d_next"].append(seg_geom(P[inx[i]], A[c], B[c])[0])
            cols["twin"].append(has_twin[c]); cols["indeg_u"].append(indeg_u[c]); cols["outdeg_v"].append(outdeg_v[c])
            cols["n_cand"].append(np.full(k, k))
    if cols["gi"]:
        X = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    else:
        X = pd.DataFrame({k: np.array([], dtype=float) for k in cols})
    gi = X.gi.values.astype(int)
    for name, arr in [("speed", speed), ("dt_prev", dt_prev), ("step_prev", step_prev), ("turn_cos", turn_cos),
                      ("hvalid", hvalid.astype(float))]:
        X[name] = arr[gi]
    X = X.astype({c: np.float32 for c in X.columns if c not in ("gi", "arc", "rank", "n_cand")})
    ctx = dict(n=n, lon0=lon0, lat0=lat0, P=P, N=N, A=A, B=B, L=L, arcs=ar, key=key, nn=nn, ts=ts)
    return X, ctx


def feature_matrix(X, feats=FEATS):
    return np.nan_to_num(X[feats].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------------------------------------------------------------------------- labeller (route needed)
def label_points(ctx, route):
    """Per-GPS-point reference arc, derived by monotone alignment of the GPS track to the ordered route.
    Returns (lab_arc per point, distance of point to that arc)."""
    P, A, B = ctx["P"], ctx["A"], ctx["B"]
    D = seg_dist(P, A[route], B[route]); path = align(D)
    return route[path], D[np.arange(len(P)), path]
