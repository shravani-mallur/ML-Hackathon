"""Folium maps for the student-test (development) trips: raw GPS, nearby road network, reference route,
classifier-only prediction and classifier+HMM prediction. Uses data/student_test_predictions.csv from hmm_experiment.py.

Outputs: maps/student_test_<trip>.html
Run from the repository root:  python src/make_maps.py [trip_id ...]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathsnap_core import MAPS, find_student_root, data_path, load_trip_inputs, load_route, trip_base, build_candidates, unproject
from route_map import build_map
import hmm

ROOT = find_student_root(); os.makedirs(MAPS, exist_ok=True)
pred = pd.read_csv(data_path("student_test_predictions.csv"), dtype={"trip": str}); pred["trip"] = pred.trip.str.zfill(8)
trips = sys.argv[1:] or sorted(pred.trip.unique())


def ll(P, ctx):
    return [[float(b), float(a)] for a, b in unproject(np.asarray(P).reshape(-1, 2), ctx["lon0"], ctx["lat0"])]


def line(seq, ctx):
    out = []
    for s, e in zip(ll(ctx["A"][seq], ctx), ll(ctx["B"][seq], ctx)):
        if not out or out[-1] != s: out.append(s)
        out.append(e)
    return out


for tid in trips:
    base = trip_base(f"{ROOT}/student_test", tid)
    X, ctx = build_candidates(*load_trip_inputs(base))
    prep = hmm.prepare(X, ctx)
    g = pred[pred.trip == tid].sort_values("gps_idx")
    hmm_route = hmm.fill_route(prep, ctx, g.hmm_arc.values)
    cls = g.classifier_arc.values; cls_seq = [int(a) for i, a in enumerate(cls) if a >= 0 and (i == 0 or a != cls[i - 1])]
    cand = np.unique(X.arc.values)
    network = [[a, b] for a, b in zip(ll(ctx["A"][cand], ctx), ll(ctx["B"][cand], ctx))]
    m = build_map(ll(ctx["P"], ctx), network=network, reference=line(list(load_route(base)), ctx),
                  classifier=line(cls_seq, ctx), predicted=line(hmm_route, ctx),
                  title=f"Student-test trip {tid}")
    out = os.path.join(MAPS, f"student_test_{tid}.html"); m.save(out)
    print(f"saved {out}")
