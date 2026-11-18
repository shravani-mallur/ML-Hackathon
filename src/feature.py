"""Step 3: build candidates + features for all train and student-test trips, and derive per-point labels.

Candidate/feature code lives in pathsnap_core.build_candidates (inference-only, no .route), so the API uses
the exact same features. Labels are added here with pathsnap_core.label_points (needs .route, training only).

Outputs (data/): features_train.pkl, features_student_test.pkl, feature_cols.json, candidate_build_summary.csv,
point_labels.csv
Run from the repository root:  python src/feature.py
"""
import os, glob, time, json, warnings
import numpy as np, pandas as pd
from pathsnap_core import (K_MAX, R_CAND, FEATS, find_student_root, data_path, load_trip_inputs, load_route,
                           trip_base, build_candidates, label_points)
warnings.filterwarnings("ignore")

SUSPECT_MEDIAN_M = 30.0                      # trip flagged if median dist(GPS, reference route) > 30 m
ROOT = find_student_root()


def build_trip(split, tid):
    base = trip_base(f"{ROOT}/{split}", tid)
    track, nodes, arcs = load_trip_inputs(base)
    X, ctx = build_candidates(track, nodes, arcs)                    # GPS + network only
    lab, d_true = label_points(ctx, load_route(base))                # reference label per GPS point (training only)
    key, n = ctx["key"], ctx["n"]
    X["y"] = (key[X.arc.values] == key[lab][X.gi.values]).astype(np.int8)
    X.insert(0, "trip", tid); X.insert(0, "split", split)

    pos = X.groupby("gi").y.max()                                     # points whose reference arc is among candidates
    nearest = X.loc[X.groupby("gi").d.idxmin()].y.sum()               # baseline 1: nearest arc
    X["_s"] = X.d - 10 * X.cos_h                                      # baseline 2: nearest with heading penalty
    heur = X.loc[X.groupby("gi")._s.idxmin()].y.sum(); X = X.drop(columns="_s")
    summ = dict(split=split, trip=tid, n_gps=n, n_rows=len(X), pts_with_pos=int(pos.sum()), pts_with_cand=int(X.gi.nunique()),
                hit_nearest=int(nearest), hit_heur=int(heur), d_true_median=float(np.median(d_true)))
    pts = pd.DataFrame(dict(split=split, trip=tid, gps_idx=np.arange(n), ts=ctx["ts"], lab_arc=lab, d_true=d_true))
    return X, summ, pts


if __name__ == "__main__":
    t0 = time.time(); Xs = {"train": [], "student_test": []}; S, PT = [], []
    for split in ["train", "student_test"]:
        for d in sorted(glob.glob(f"{ROOT}/{split}/*")):
            tid = os.path.basename(d); X, s, p = build_trip(split, tid); Xs[split].append(X); S.append(s); PT.append(p)
            print(f"{split}/{tid}: {s['n_gps']} pts, {s['n_rows']} rows ({time.time()-t0:.0f}s)")

    S = pd.DataFrame(S); S["suspect"] = S.d_true_median > SUSPECT_MEDIAN_M
    S.to_csv(data_path("candidate_build_summary.csv"), index=False); pd.concat(PT).to_csv(data_path("point_labels.csv"), index=False)
    json.dump(FEATS, open(data_path("feature_cols.json"), "w"))
    sus = set(S[S.suspect].trip)
    for split in Xs:
        df = pd.concat(Xs[split], ignore_index=True); df["suspect"] = df.trip.isin(sus); df.to_pickle(data_path(f"features_{split}.pkl"))
        print(f"saved features_{split}.pkl  shape={df.shape}  positive rate={df.y.mean():.3f}")

    print("\n=== CANDIDATE RULE: nearest %d arcs within %d m ===" % (K_MAX, R_CAND))
    for split in ["train", "student_test"]:
        for label, sub in [("all trips", S[S.split == split]), ("excluding suspect trips", S[(S.split == split) & ~S.suspect])]:
            n = sub.n_gps.sum()
            print(f"{split:13s} {label:24s} miss rate={100*(1-sub.pts_with_pos.sum()/n):5.2f}%  no-candidate pts={100*(1-sub.pts_with_cand.sum()/n):5.2f}%  "
                  f"| baseline nearest-arc top-1={100*sub.hit_nearest.sum()/n:5.1f}%  nearest+heading={100*sub.hit_heur.sum()/n:5.1f}%")
    print("\nSuspect trips (median GPS->reference distance > %.0f m):" % SUSPECT_MEDIAN_M, sorted(sus))
    S["miss_rate"] = 1 - S.pts_with_pos / S.n_gps
    print("\nWorst 8 trips by miss rate:\n", S.sort_values("miss_rate", ascending=False).head(8)[["split", "trip", "n_gps", "miss_rate", "d_true_median"]].to_string())
    print("Feature columns:", FEATS, "| total time %.0fs" % (time.time() - t0))
