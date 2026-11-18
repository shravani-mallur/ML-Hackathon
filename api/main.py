"""PathSnap FastAPI backend (Laptop 1).

Hosts the tuned classifier, the HMM/Viterbi decoder, the road networks of the evaluation trips and the server-side
evaluation. The UI only sends a trip id and receives the prediction + metrics; reference routes never leave this
process (the response contains metrics computed from them, never the reference arcs or geometry).

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000      (from the repository root)
Env:  PATHSNAP_EVAL_DIR  folder with <trip_id>/<trip_id>.{track,nodes,arcs,route}  (default: server_data/evaluator_test)
"""
import os, sys, json, time, logging
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))   # avoids joblib's wmic warning on Windows 11
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from pathsnap_core import (MODELS, load_trip_inputs, load_route, trip_base, build_candidates, feature_matrix,
                           label_points, unproject)
import hmm

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(REPO, "server_data", "evaluator_test"))
LOG_DIR = os.path.join(REPO, "api", "logs"); os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(LOG_DIR, "api.log"))])
log = logging.getLogger("pathsnap")

# ------------------------------------------------------------------ model artifacts (loaded once at start-up)
MODEL = joblib.load(os.path.join(MODELS, "histgb_tuned.joblib"))
FEATS = json.load(open(os.path.join(MODELS, "feature_cols.json")))
HMM_CFG = json.load(open(os.path.join(MODELS, "hmm_config.json")))
HMM_PARAMS = HMM_CFG["params"]
log.info("Loaded model %s with %d features; HMM params %s", type(MODEL).__name__, len(FEATS), HMM_PARAMS)
log.info("Evaluation trip folder: %s", EVAL_DIR)

app = FastAPI(title="PathSnap API", version="1.0",
              description="Map-matching: tuned HistGradientBoosting candidate classifier + HMM/Viterbi decoding")


def list_trips():
    if not os.path.isdir(EVAL_DIR):
        return []
    return sorted(t for t in os.listdir(EVAL_DIR)
                  if all(os.path.isfile(trip_base(EVAL_DIR, t) + e) for e in (".track", ".nodes", ".arcs")))


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path,
             request.client.host if request.client else "?", response.status_code, (time.time() - t0) * 1000)
    return response


@app.get("/")
def root():
    return dict(service="PathSnap API", endpoints=["/health", "/trips", "POST /predict", "/docs"],
                ui="Run ui/app.py on Laptop 2 and open http://localhost:8501")


@app.get("/health")
def health():
    return dict(status="ok", model=type(MODEL).__name__, n_features=len(FEATS), n_trips=len(list_trips()))


@app.get("/trips")
def trips():
    return dict(trips=list_trips())


def predict_trip(base):
    """Full prediction pipeline. Reads ONLY .track/.nodes/.arcs."""
    timings = {}
    t0 = time.time(); track, nodes, arcs = load_trip_inputs(base); timings["load_s"] = time.time() - t0
    t0 = time.time(); X, ctx = build_candidates(track, nodes, arcs); timings["features_s"] = time.time() - t0
    t0 = time.time(); prob = MODEL.predict_proba(feature_matrix(X, FEATS))[:, 1] if len(X) else np.zeros(0)
    timings["classifier_s"] = time.time() - t0
    t0 = time.time(); prep = hmm.prepare(X, ctx)
    rows_hmm = hmm.decode(prep, prob, HMM_PARAMS); rows_cls = hmm.argmax_rows(prep, prob)
    timings["hmm_s"] = time.time() - t0
    return X, ctx, prob, prep, rows_hmm, rows_cls, timings


def score(base, ctx, prep, rows_hmm, rows_cls):
    """Server-side evaluation. The reference route is read here, AFTER prediction, and only metrics are returned."""
    route = load_route(base)
    lab, _ = label_points(ctx, route)
    key = ctx["key"]; out = {}
    for name, rows in [("hmm", rows_hmm), ("classifier_only", rows_cls)]:
        pa = hmm.per_point_arcs(prep, rows)
        hit = (pa >= 0) & (key[np.maximum(pa, 0)] == key[lab])
        prec, rec, f1 = hmm.route_overlap(pa, route, ctx)
        out[name] = dict(top1_accuracy=round(float(hit.mean()), 4), correct_points=int(hit.sum()), n_points=int(len(pa)),
                         route_continuity=round(hmm.continuity(pa, ctx["arcs"]), 4), route_precision=round(prec, 4),
                         route_recall=round(rec, 4), route_f1=round(f1, 4))
    return out


def latlon(P, ctx):
    ll = unproject(np.asarray(P, float).reshape(-1, 2), ctx["lon0"], ctx["lat0"])
    return [[round(float(a), 6), round(float(b), 6)] for b, a in ll]          # -> [lat, lon]


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if tid not in list_trips():
        raise HTTPException(status_code=404, detail=f"Unknown trip id {tid!r}")
    base = trip_base(EVAL_DIR, tid); t_all = time.time()
    X, ctx, prob, prep, rows_hmm, rows_cls, timings = predict_trip(base)
    metrics = score(base, ctx, prep, rows_hmm, rows_cls)
    timings["total_s"] = time.time() - t_all

    A, B, ar = ctx["A"], ctx["B"], ctx["arcs"]
    pa = hmm.per_point_arcs(prep, rows_hmm); pa_cls = hmm.per_point_arcs(prep, rows_cls)
    snapped = np.full((ctx["n"], 2), np.nan)                                  # GPS point projected onto its arc
    if len(rows_hmm):
        tt = X.t.values[rows_hmm][:, None]; a = X.arc.values[rows_hmm]
        snapped[prep["pts"]] = A[a] + tt * (B[a] - A[a])
    gps_ll = latlon(ctx["P"], ctx)
    snap_ll = latlon(np.nan_to_num(snapped), ctx)
    points = [dict(idx=i, lat=gps_ll[i][0], lon=gps_ll[i][1], arc_id=int(pa[i]),
                   snapped=None if pa[i] < 0 else snap_ll[i]) for i in range(ctx["n"])]
    route_arcs = hmm.fill_route(prep, ctx, pa)

    def arcs_to_line(seq):
        line = []
        if not seq: return line
        sa, sb = latlon(A[seq], ctx), latlon(B[seq], ctx)
        for s, e in zip(sa, sb):
            if not line or line[-1] != s: line.append(s)
            line.append(e)
        return line

    cand_arcs = np.unique(X.arc.values) if len(X) else np.zeros(0, int)
    na, nb = latlon(A[cand_arcs], ctx), latlon(B[cand_arcs], ctx)
    network = [[x, y] for x, y in zip(na, nb)]
    cls_seq = [int(a) for i, a in enumerate(pa_cls) if a >= 0 and (i == 0 or a != pa_cls[i - 1])]
    log.info("trip %s: %d GPS points, %d candidate rows, HMM top-1 %.4f (classifier-only %.4f), %.2fs",
             tid, ctx["n"], len(X), metrics["hmm"]["top1_accuracy"], metrics["classifier_only"]["top1_accuracy"],
             timings["total_s"])
    return dict(trip_id=tid, n_points=ctx["n"], n_candidate_rows=int(len(X)),
                unmatched_points=int((pa < 0).sum()), points=points,
                predicted_route_arcs=route_arcs, predicted_route=arcs_to_line(route_arcs),
                classifier_only_route=arcs_to_line(cls_seq), network=network,
                metrics=metrics, timings={k: round(v, 3) for k, v in timings.items()},
                model=dict(classifier="HistGradientBoosting (Optuna-tuned)", decoder="HMM + Viterbi", hmm_params=HMM_PARAMS))
