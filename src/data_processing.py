"""Step 1: raw-data inventory and checks (train + student-test partitions only).

For every trip: file availability, GPS row count, missing (NaN) values, duplicate / negative timestamp steps,
share of 1-second steps, largest time gap, coordinate extents, node / arc counts, self-loops, route length,
road-network connectivity (weakly connected components, isolated nodes), and relationship checks
(arc node ids < number of nodes, route arc ids < number of arcs).

Outputs (data/): dataset_inventory.csv, network_connectivity.csv
Run from the repository root:  python src/data_processing.py
"""
import os, sys, glob
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathsnap_core import find_student_root, data_path, read_table

ROOT = find_student_root()
inv, conn = [], []
for split in ["train", "student_test"]:
    for d in sorted(glob.glob(os.path.join(ROOT, split, "*"))):
        tid = os.path.basename(d); base = os.path.join(d, tid)
        exts = sorted(f.split(".")[-1] for f in os.listdir(d))
        t = pd.read_csv(base + ".track", sep=r"\s+", header=None).values.astype(float)
        nd = read_table(base + ".nodes").astype(float); ar = read_table(base + ".arcs").astype(np.int64)
        rt = read_table(base + ".route")[:, 0].astype(np.int64)
        dts = np.diff(t[:, 2])
        inv.append(dict(trip=os.path.join(split, tid, tid), split=split, exts=",".join(exts), n_gps=len(t),
                        nan=int(np.isnan(t).sum()), dup_ts=int((dts == 0).sum()), neg_ts=int((dts < 0).sum()),
                        pct_1s=round(100 * float((dts == 1).mean()), 1), max_gap_s=float(dts.max()) if len(dts) else 0.0,
                        lon_min=t[:, 0].min(), lon_max=t[:, 0].max(), lat_min=t[:, 1].min(), lat_max=t[:, 1].max(),
                        n_nodes=len(nd), node_lon_min=nd[:, 0].min(), node_lon_max=nd[:, 0].max(),
                        node_lat_min=nd[:, 1].min(), node_lat_max=nd[:, 1].max(), n_arcs=len(ar),
                        arc_max_node=int(ar.max()), self_loops=int((ar[:, 0] == ar[:, 1]).sum()), n_route=len(rt),
                        route_unique=len(np.unique(rt)), route_max_arc=int(rt.max())))
        nn = len(nd)
        G = csr_matrix((np.ones(len(ar)), (ar[:, 0], ar[:, 1])), shape=(nn, nn))
        ncomp, lab = connected_components(G, directed=True, connection="weak")
        deg = np.bincount(ar.ravel(), minlength=nn)
        route_comp = np.unique(lab[ar[rt, 0]])
        conn.append(dict(split=split, trip=tid, n_nodes=nn, n_arcs=len(ar), weak_components=int(ncomp),
                         largest_component_pct=round(100 * np.bincount(lab).max() / nn, 2), isolated_nodes=int((deg == 0).sum()),
                         arcs_valid_node_ids=bool(ar.max() < nn and ar.min() >= 0), route_valid_arc_ids=bool(rt.max() < len(ar)),
                         route_in_one_component=bool(len(route_comp) == 1)))
        print(f"{split}/{tid}: {len(t)} GPS, {nn} nodes, {len(ar)} arcs, {len(rt)} route arcs, {ncomp} components")

I = pd.DataFrame(inv).round({c: 5 for c in ["lon_min", "lon_max", "lat_min", "lat_max", "node_lon_min", "node_lon_max", "node_lat_min", "node_lat_max"]}); C = pd.DataFrame(conn)
I.to_csv(data_path("dataset_inventory.csv"), index=False); C.to_csv(data_path("network_connectivity.csv"), index=False)
print("\nTrips per partition:\n", I.split.value_counts().to_string())
print("\nMissing values (NaN) in .track files:", int(I.nan.sum()), "| duplicate timestamps:", int(I.dup_ts.sum()),
      "| negative steps:", int(I.neg_ts.sum()))
print("Trips with a time gap > 60 s:", int((I.max_gap_s > 60).sum()))
print("Networks with isolated nodes:", int((C.isolated_nodes > 0).sum()), "| all node/arc ids valid:",
      bool(C.arcs_valid_node_ids.all() and C.route_valid_arc_ids.all()))
