"""HMM + Viterbi route reconstruction on top of the candidate classifier.

Hidden states   : for GPS point i, the states are its candidate arcs (rows of X with gi == i), so the
                  candidate -> road-segment correspondence is preserved (state k of point i IS arc X.arc[k]).
Emission        : log P(candidate is the reference arc | features) from the tuned classifier, renormalised over
                  the point's candidates, multiplied by a weight w_emis.
Transition      : Newson & Krumm style. For previous state a (position ta along arc a) and next state b (tb):
                    route distance r = network distance travelled from (a, ta) to (b, tb)
                                       (same arc: |tb-ta|*L ; else (1-ta)*La + shortest_path(a.to, b.from) + tb*Lb)
                    g = straight-line distance between the two GPS points (spatial, NOT elapsed time)
                    log T = -|r - g| / beta   - uturn_pen (if b is the reverse twin of a)
                                              - back_pen  (if moving backwards > BACK_TOL m along the same arc)
                  If b is not reachable from a within the search limit, the transition is "disconnected" and gets a
                  constant log-penalty -disc_pen, so the decoder can still jump (it never gets stuck).
Missing points  : GPS points without any candidate are skipped (no state); the chain links the neighbouring points
                  with candidates and their GPS distance. They are reported as unmatched (-1).
GPS gaps        : if the GPS distance between consecutive matched points exceeds gap_m, the chain is broken there
                  (uniform transition), i.e. the decoder restarts and relies on emissions only.
Viterbi         : delta_i(b) = emis_i(b) + max_a [delta_{i-1}(a) + logT(a, b)], with back-pointers; then backtrack.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

DEFAULT_PARAMS = dict(w_emis=1.0, beta=10.0, disc_pen=12.0, uturn_pen=4.0, back_pen=2.0, gap_m=1000.0)
BACK_TOL = 10.0          # metres of backwards movement on the same arc tolerated as GPS noise
CHUNK = 40               # points per Dijkstra batch (memory control)
MAX_LIMIT = 4000.0       # cap on the shortest-path search radius in metres


def road_graph(ctx):
    """Directed graph over nodes, weight = arc length (duplicates collapsed to the shortest)."""
    ar, L, nn = ctx["arcs"], np.maximum(ctx["L"], 0.01), ctx["nn"]
    df = pd.DataFrame(dict(u=ar[:, 0], v=ar[:, 1], w=L)).groupby(["u", "v"], as_index=False).w.min()
    return csr_matrix((df.w.values, (df.u.values, df.v.values)), shape=(nn, nn))


def prepare(X, ctx):
    """Precompute everything about the trip that does not depend on decoder parameters.

    Returns a dict with, for every matched point, its candidate rows, and for every consecutive pair of matched
    points the matrices r (route distance, inf if disconnected), uturn, back and the scalar GPS distance g.
    """
    ar, L, P = ctx["arcs"], ctx["L"], ctx["P"]
    gi = X.gi.values.astype(int)
    order = np.lexsort((X["rank"].values, gi)); gi_s = gi[order]
    pts, starts = np.unique(gi_s, return_index=True)
    ends = np.r_[starts[1:], len(gi_s)]
    rows = [order[s:e] for s, e in zip(starts, ends)]                        # row indices of X per matched point
    arc = X.arc.values.astype(int); tt = X.t.values.astype(float)
    G = road_graph(ctx)
    steps = []
    gdist = np.linalg.norm(P[pts[1:]] - P[pts[:-1]], axis=1) if len(pts) > 1 else np.zeros(0)
    for c0 in range(1, len(pts), CHUNK):                                      # batch Dijkstra over CHUNK transitions
        c1 = min(c0 + CHUNK, len(pts))
        src = np.unique(np.concatenate([ar[arc[rows[i - 1]], 1] for i in range(c0, c1)]))
        lim = float(min(MAX_LIMIT, 2.0 * gdist[c0 - 1:c1 - 1].max() + 300.0))
        Dm = dijkstra(G, directed=True, indices=src, limit=lim)
        pos = {s: k for k, s in enumerate(src)}
        for i in range(c0, c1):
            ra, rb = rows[i - 1], rows[i]
            a, b = arc[ra], arc[rb]; ta, tb = tt[ra], tt[rb]
            sp = Dm[[pos[s] for s in ar[a, 1]]][:, ar[b, 0]]                 # shortest path a.to -> b.from
            r = (1 - ta)[:, None] * L[a][:, None] + sp + (tb * L[b])[None, :]
            same = (ar[a][:, None, :] == ar[b][None, :, :]).all(2)
            dpos = (tb[None, :] * L[b][None, :]) - (ta * L[a])[:, None]       # signed movement along the same arc
            r = np.where(same, np.abs(dpos), r)
            back = same & (dpos < -BACK_TOL)
            uturn = (ar[a, 1][:, None] == ar[b, 0][None, :]) & (ar[a, 0][:, None] == ar[b, 1][None, :])
            steps.append(dict(r=r, back=back, uturn=uturn, g=float(gdist[i - 1])))
    return dict(pts=pts, rows=rows, steps=steps, n=ctx["n"], arc=arc, G=G)


def decode(prep, prob, params=None):
    """Viterbi over prepared trip. prob = classifier probability per row of X. Returns chosen row per matched point."""
    q = dict(DEFAULT_PARAMS, **(params or {}))
    rows, steps = prep["rows"], prep["steps"]
    if not rows:
        return np.zeros(0, int)
    eps = 1e-6

    def emis(rw):
        p = np.clip(prob[rw], eps, 1.0); p = p / p.sum()
        return q["w_emis"] * np.log(p)

    delta = emis(rows[0]); back = []
    for i in range(1, len(rows)):
        s = steps[i - 1]
        if s["g"] > q["gap_m"]:                                              # large GPS gap: break the chain
            T = np.zeros_like(s["r"])
        else:
            T = -np.abs(s["r"] - s["g"]) / q["beta"] - q["uturn_pen"] * s["uturn"] - q["back_pen"] * s["back"]
            T = np.where(np.isfinite(s["r"]), T, -q["disc_pen"])            # disconnected / unreachable
        M = delta[:, None] + T
        bp = M.argmax(0); back.append(bp)
        delta = M[bp, np.arange(M.shape[1])] + emis(rows[i])
    k = int(delta.argmax()); path = [k]
    for bp in reversed(back):
        k = int(bp[k]); path.append(k)
    path = path[::-1]
    return np.array([rows[i][k] for i, k in enumerate(path)])


def argmax_rows(prep, prob):
    """Classifier-only baseline: highest-probability candidate per matched point."""
    return np.array([rw[np.argmax(prob[rw])] for rw in prep["rows"]])


def per_point_arcs(prep, chosen_rows):
    """Arc id per GPS point (length n), -1 where the point had no candidates."""
    out = np.full(prep["n"], -1, dtype=np.int64)
    if len(chosen_rows):
        out[prep["pts"]] = prep["arc"][chosen_rows]
    return out


def fill_route(prep, ctx, point_arcs):
    """Collapse per-point arcs into a connected arc sequence, inserting shortest-path arcs between consecutive
    decoded arcs that are not directly connected (for drawing the route). Returns list of arc ids."""
    ar, G = ctx["arcs"], prep["G"]
    seq = [a for a in point_arcs if a >= 0]
    route = []
    for a in seq:
        if route and (ar[route[-1]] == ar[a]).all():
            continue
        if route:
            u, v = ar[route[-1], 1], ar[a, 0]
            if u != v:
                d, pred = dijkstra(G, directed=True, indices=u, limit=MAX_LIMIT, return_predecessors=True)
                if np.isfinite(d[v]):
                    path_nodes = [v]
                    while path_nodes[-1] != u:
                        path_nodes.append(pred[path_nodes[-1]])
                    path_nodes = path_nodes[::-1]
                    lut = {(int(x), int(y)): k for k, (x, y) in enumerate(ar)}
                    route += [lut[(int(x), int(y))] for x, y in zip(path_nodes[:-1], path_nodes[1:])
                              if (int(x), int(y)) in lut]
        route.append(int(a))
    return route


# ----------------------------------------------------------------------------------------------- metrics
def continuity(arcs_seq, ar):
    """Fraction of consecutive predicted arcs that are the same arc or connected (end node -> start node)."""
    a = np.asarray([x for x in arcs_seq if x >= 0])
    if len(a) < 2:
        return 1.0
    same = (ar[a[:-1]] == ar[a[1:]]).all(1); conn = ar[a[:-1], 1] == ar[a[1:], 0]
    return float((same | conn).mean())


def route_overlap(pred_point_arcs, ref_route, ctx):
    """Sequence-level: length-weighted precision / recall of the set of traversed road segments (by node pair)."""
    ar, L, nn = ctx["arcs"], ctx["L"], ctx["nn"]
    key = ar[:, 0] * nn + ar[:, 1]
    pk = {int(key[a]): L[a] for a in pred_point_arcs if a >= 0}
    rk = {int(key[a]): L[a] for a in ref_route}
    inter = set(pk) & set(rk)
    prec = sum(pk[k] for k in inter) / max(sum(pk.values()), 1e-9)
    rec = sum(rk[k] for k in inter) / max(sum(rk.values()), 1e-9)
    return prec, rec, (2 * prec * rec / max(prec + rec, 1e-9))
