import os, glob, time, warnings
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")

SEED, RMAX, N_SAMPLE = 42, 200.0, 300      # N_SAMPLE: GPS points sampled per trip for candidate analysis
rng = np.random.default_rng(SEED)

ROOT = next((r for r, d, _ in os.walk(".") if "train" in d and "student_test" in d), None)
assert ROOT, "Set ROOT manually to the folder containing train/ and student_test/"
print("ROOT:", os.path.abspath(ROOT))

def load_trip(base):
    rd = lambda ext: pd.read_csv(base + ext, sep=r"\s+", header=None).values
    return (rd(".track").astype(float), rd(".nodes").astype(float),
            rd(".arcs").astype(np.int64), rd(".route")[:, 0].astype(np.int64))

def project(xy, lon0, lat0):           # local equirectangular -> metres
    kx = 111320 * np.cos(np.deg2rad(lat0)); ky = 110540.0
    return np.c_[(xy[:, 0] - lon0) * kx, (xy[:, 1] - lat0) * ky]

def seg_dist(P, A, B):                 # point-to-segment distance, (n,m) float32 metres
    AB = B - A; L2 = np.maximum((AB ** 2).sum(1), 1e-9)
    out = np.empty((len(P), len(A)), np.float32)
    for s in range(0, len(P), 2000):
        p = P[s:s + 2000, None, :]
        t = np.clip(((p - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
        proj = A[None] + t[..., None] * AB[None]
        out[s:s + 2000] = np.sqrt(((p - proj) ** 2).sum(2))
    return out

def align(D):
    """Monotone alignment: GPS point i -> route position j(i), j non-decreasing, minimising total distance."""
    n, m = D.shape; idx = np.arange(m)
    cost = D[0].astype(np.float64); back = np.empty((n, m), np.int32); back[0] = idx
    for i in range(1, n):
        macc = np.minimum.accumulate(cost)
        back[i] = np.maximum.accumulate(np.where(cost == macc, idx, 0))
        cost = D[i] + macc
    j = int(cost.argmin()); path = np.empty(n, np.int32)
    for i in range(n - 1, -1, -1):
        path[i] = j; j = back[i][j]
    return path

trips = [(s, os.path.basename(d)) for s in ["train", "student_test"]
         for d in sorted(glob.glob(f"{ROOT}/{s}/*"))]
rows, recs, t0 = [], [], time.time()

for k, (split, tid) in enumerate(trips):
    t, nd, ar, rt = load_trip(f"{ROOT}/{split}/{tid}/{tid}")
    lon0, lat0 = t[:, 0].mean(), t[:, 1].mean()
    P = project(t[:, :2], lon0, lat0); N = project(nd, lon0, lat0)
    A, B = N[ar[:, 0]], N[ar[:, 1]]; arc_len = np.linalg.norm(B - A, axis=1)

    dts = np.diff(t[:, 2]); step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    speed = step / np.maximum(dts, 1e-9)
    cont = float((ar[rt[:-1], 1] == ar[rt[1:], 0]).mean()) if len(rt) > 1 else 1.0

    # per-point labels by aligning GPS to the ordered route
    D = seg_dist(P, A[rt], B[rt]); path = align(D)
    lab = rt[path]; d_true = D[np.arange(len(P)), path]; del D

    # heading check: does GPS motion agree with the labelled arc's direction?
    hv = np.zeros_like(P); hv[1:-1] = P[2:] - P[:-2]
    av = (B - A)[lab]; nh, na = np.linalg.norm(hv, axis=1), np.linalg.norm(av, axis=1)
    ok = (nh > 3) & (na > 1)
    head_pos = float((((hv[ok] * av[ok]).sum(1) / (nh[ok] * na[ok])) > 0).mean()) if ok.any() else np.nan

    key = ar[:, 0] * len(nd) + ar[:, 1]; rkey = ar[:, 1] * len(nd) + ar[:, 0]
    rows.append(dict(split=split, trip=tid, n_gps=len(P), n_arcs=len(ar), n_route=len(rt),
        route_chain_cont=round(cont, 4), route_repeats=len(rt) - len(set(rt)),
        gaps_gt5s=int((dts > 5).sum()), max_gap_s=float(dts.max()), jump_at_max_gap_m=round(float(step[dts.argmax()]), 1),
        speed_p99_mps=round(float(np.percentile(speed, 99)), 1), n_speed_gt50=int((speed > 50).sum()),
        arc_len_median_m=round(float(np.median(arc_len)), 1), twin_arc_pct=round(float(np.isin(rkey, key).mean()) * 100, 1),
        dup_arcs=int(pd.Series(key).duplicated().sum()), self_loops=int((ar[:, 0] == ar[:, 1]).sum()),
        d_true_median=round(float(np.median(d_true)), 1), d_true_p95=round(float(np.percentile(d_true, 95)), 1),
        d_true_max=round(float(d_true.max()), 1), heading_agree_pct=round(head_pos * 100, 1)))

    # candidate index: sample points ~every 20 m along every arc
    ns = np.maximum(2, np.ceil(arc_len / 20).astype(int) + 1)
    arc_idx = np.repeat(np.arange(len(ar)), ns)
    kk = np.arange(ns.sum()) - np.repeat(np.cumsum(ns) - ns, ns)
    S = A[arc_idx] + (kk / np.repeat(ns - 1, ns))[:, None] * (B - A)[arc_idx]
    tree = cKDTree(S)
    sel = np.sort(rng.choice(len(P), size=min(len(P), N_SAMPLE), replace=False))
    for i, nb in zip(sel, tree.query_ball_point(P[sel], r=RMAX + 12)):
        cand = np.unique(arc_idx[np.asarray(nb, dtype=int)]) if len(nb) else np.array([], dtype=int)
        dd = seg_dist(P[i:i + 1], A[cand], B[cand])[0] if len(cand) else np.array([])
        rank = int((dd <= d_true[i] + 1e-3).sum()) if d_true[i] <= RMAX else np.nan
        recs.append((split, tid, int(i), float(d_true[i]), rank, int((dd <= 50).sum()),
                     int((dd <= 100).sum()), int((dd <= RMAX).sum())))
    if k % 10 == 0: print(f"[{k+1}/{len(trips)}] {split}/{tid}  {time.time()-t0:.0f}s")

T = pd.DataFrame(rows)
Q = pd.DataFrame(recs, columns=["split", "trip", "gps_idx", "d_true", "rank_true", "n_cand50", "n_cand100", "n_cand200"])
T.to_csv("trip_stats.csv", index=False); Q.to_csv("candidate_sample.csv", index=False)

print("\n=== SUMMARY (decisions below use TRAIN rows only) ===")
tr = Q[Q.split == "train"]
print("True-arc distance percentiles 50/75/90/95/99 (m):", np.percentile(tr.d_true, [50, 75, 90, 95, 99]).round(1))
print("\nCandidate miss rate by RADIUS (% of GPS points whose true arc is farther than R):")
for R in [20, 30, 50, 75, 100, 150, 200]:
    print(f"  R={R:>3} m : {(tr.d_true > R).mean()*100:5.2f}%")
print("\nCandidate miss rate by TOP-K nearest arcs (ties/twins counted):")
for K in [3, 5, 8, 10, 15, 20, 30]:
    print(f"  K={K:>2} : {((tr.rank_true.isna()) | (tr.rank_true > K)).mean()*100:5.2f}%")
print("\nMean candidates within 50 m / 100 m / 200 m:", tr.n_cand50.mean().round(1), tr.n_cand100.mean().round(1), tr.n_cand200.mean().round(1))
print("\nStudent-test miss rate at R=50 / 100 m:", *[(Q[Q.split=='student_test'].d_true > R).mean().round(4) for R in (50, 100)])
print("\n=== TRIP-LEVEL CHECKS (all 90) ===")
print("Route chains not fully connected:\n", T[T.route_chain_cont < 1][["split", "trip", "route_chain_cont"]].to_string())
print("\nWorst time gaps:\n", T.sort_values("max_gap_s", ascending=False).head(8)[["split", "trip", "n_gps", "gaps_gt5s", "max_gap_s", "jump_at_max_gap_m"]].to_string())
print("\nTrips with implausible speeds (>50 m/s steps):\n", T[T.n_speed_gt50 > 0][["trip", "n_speed_gt50", "speed_p99_mps"]].to_string())
print("\nHeading agreement of labelled arcs (%):", T.heading_agree_pct.describe().round(1).to_dict())
print("Twin-arc %:", T.twin_arc_pct.describe()[["min", "mean"]].round(1).to_dict(),
      "| duplicate arcs total:", int(T.dup_arcs.sum()), "| self loops total:", int(T.self_loops.sum()))
print("Trips where median true-arc distance > 30 m (possible label/alignment issue):\n",
      T[T.d_true_median > 30][["trip", "d_true_median", "d_true_p95"]].to_string())

# ---------- Folium map for Review 1 (small train trip; network + GPS + reference route) ----------
def make_map(split, tid, out):
    import folium
    t, nd, ar, rt = load_trip(f"{ROOT}/{split}/{tid}/{tid}")
    m = folium.Map(location=[t[:, 1].mean(), t[:, 0].mean()], zoom_start=14)
    net, rte, gps = (folium.FeatureGroup(n) for n in ["Road network (grey)", "Reference route (green)", "Raw GPS (blue)"])
    for u, v in ar:
        folium.PolyLine([(nd[u, 1], nd[u, 0]), (nd[v, 1], nd[v, 0])], color="gray", weight=1, opacity=.5).add_to(net)
    for a in rt:
        u, v = ar[a]
        folium.PolyLine([(nd[u, 1], nd[u, 0]), (nd[v, 1], nd[v, 0])], color="green", weight=5).add_to(rte)
    for i in range(0, len(t), 2):
        folium.CircleMarker((t[i, 1], t[i, 0]), radius=2, color="blue", fill=True, tooltip=f"t={t[i,2]:.0f}s").add_to(gps)
    for g in (net, rte, gps): g.add_to(m)
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px;border:1px solid #888">'
        '<b>Legend</b><br><span style="color:gray">&#9632;</span> road network<br>'
        '<span style="color:green">&#9632;</span> reference route<br><span style="color:blue">&#9679;</span> raw GPS</div>'))
    folium.LayerControl().add_to(m)
    m.fit_bounds([[t[:, 1].min(), t[:, 0].min()], [t[:, 1].max(), t[:, 0].max()]]); m.save(out)

try:
    make_map("train", "00000046", "eda_map_train_00000046.html"); print("\nSaved eda_map_train_00000046.html")
except Exception as e:
    print("Folium map failed:", e)
print(f"Done in {time.time()-t0:.0f}s")