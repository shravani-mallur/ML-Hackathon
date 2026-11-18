"""Step 6: final tuned classifier + HMM/Viterbi decoder tuning and evaluation.

1. Out-of-fold (OOF) probabilities on the 78 non-suspect train trips (4-fold GroupKFold by trip, tuned HistGB,
   same 300k whole-point sampling protocol as models.py / tune_histgb.py). OOF is needed so that the decoder is tuned
   on emissions that look like real unseen-trip predictions, not over-confident in-sample ones.
2. Final model: tuned HistGB refit on a 1M-row whole-point sample of all 78 train trips -> models/histgb_tuned.joblib.
3. Decoder parameters tuned with Optuna TPE on the train OOF emissions ONLY (objective: pooled Top-1 over train points).
4. Student-test comparison: classifier-only (argmax per point) vs classifier + HMM (default and tuned parameters).
Evaluator-test trips are never read by this script.

Outputs (data/): hmm_decoder_trials.csv, hmm_vs_classifier.csv, hmm_per_trip.csv, student_test_predictions.csv,
                 final_model_metrics.json ; (models/): histgb_tuned.joblib, hmm_config.json
Run from the repository root:  python src/hmm_experiment.py
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd, joblib, optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathsnap_core import (FEATS, MODELS, find_student_root, data_path, load_trip_inputs, load_route, trip_base,
                           build_candidates, feature_matrix)
import hmm
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED, N_FIT, N_FIT_FINAL, N_FOLDS, N_TRIALS = 42, 300_000, 1_000_000, 4, 30
ROOT = find_student_root(); os.makedirs(MODELS, exist_ok=True)
BEST = json.load(open(data_path("optuna_histgb_best.json")))["best_params"]


def sample_points(df, n_rows, seed):                 # identical to models.py: sample whole GPS points
    if len(df) <= n_rows: return df
    g = df.groupby(["trip", "gi"], sort=False).ngroup().values
    keep = np.random.default_rng(seed).random(g.max() + 1) < n_rows / len(df)
    return df[keep[g]]


def make_model():
    return HistGradientBoostingClassifier(random_state=SEED, **BEST)


t_all = time.time()
tr = pd.read_pickle(data_path("features_train.pkl")); te = pd.read_pickle(data_path("features_student_test.pkl"))
tr["trip"] = tr.trip.astype(str); te["trip"] = te.trip.astype(str)
tr = tr[~tr.suspect].reset_index(drop=True)          # same exclusion as model comparison (trips 33, 54)
print(f"train rows {len(tr):,} ({tr.trip.nunique()} trips) | student-test rows {len(te):,} ({te.trip.nunique()} trips)")

# ---------------------------------------------------------------- 1. OOF probabilities on train
tr["p"] = np.nan
for k, (a, b) in enumerate(GroupKFold(n_splits=N_FOLDS).split(tr, groups=tr.trip.values)):
    fit = sample_points(tr.iloc[a], N_FIT, SEED + k)
    m = make_model().fit(feature_matrix(fit), fit.y.values)
    tr.loc[tr.index[b], "p"] = m.predict_proba(feature_matrix(tr.iloc[b]))[:, 1]
    print(f"OOF fold {k} done ({time.time()-t_all:.0f}s)")

# ---------------------------------------------------------------- 2. final model (exported for the API)
fit = sample_points(tr, N_FIT_FINAL, SEED); t0 = time.time()
final = make_model().fit(feature_matrix(fit), fit.y.values); fit_s = time.time() - t0
t0 = time.time(); te["p"] = final.predict_proba(feature_matrix(te))[:, 1]; inf_ms = (time.time() - t0) / len(te) * 1e6
joblib.dump(final, os.path.join(MODELS, "histgb_tuned.joblib"))
json.dump(FEATS, open(os.path.join(MODELS, "feature_cols.json"), "w"))
print(f"final model: {len(fit):,} rows, fit {fit_s:.1f}s, test log loss {log_loss(te.y, np.clip(te.p, 1e-6, 1-1e-6)):.4f}")


# ---------------------------------------------------------------- prepare trips (geometry + transitions, once)
def prepare_split(df, split):
    out = {}
    for tid, g in df.groupby("trip", sort=True):
        base = trip_base(f"{ROOT}/{split}", tid)
        X, ctx = build_candidates(*load_trip_inputs(base))
        assert len(X) == len(g) and (X.arc.values == g.arc.values).all() and (X.gi.values == g.gi.values).all()
        lab_key = None
        pl = PL[(PL.split == split) & (PL.trip == tid)].sort_values("gps_idx")
        lab_key = ctx["key"][pl.lab_arc.values]
        out[tid] = dict(prep=hmm.prepare(X, ctx), ctx=ctx, p=g.p.values, y=g.y.values, lab_key=lab_key,
                        route=load_route(base))
    return out


PL = pd.read_csv(data_path("point_labels.csv"), dtype={"trip": str}); PL["trip"] = PL.trip.str.zfill(8)
t0 = time.time(); TR = prepare_split(tr, "train"); TE = prepare_split(te, "student_test")
print(f"prepared {len(TR)} train + {len(TE)} student-test trips in {time.time()-t0:.0f}s")


def evaluate_trip(d, rows):
    """Metrics for one trip given the chosen candidate row per matched point."""
    prep, ctx = d["prep"], d["ctx"]
    pa = hmm.per_point_arcs(prep, rows)
    hit = (pa >= 0) & (ctx["key"][np.maximum(pa, 0)] == d["lab_key"])
    sel = np.zeros(len(d["y"]), np.int8); sel[rows] = 1                 # chosen candidate = predicted positive
    prec, rec, f1r = hmm.route_overlap(pa, d["route"], ctx)
    return dict(n_pts=len(pa), hits=int(hit.sum()), top1=float(hit.mean()), unmatched=int((pa < 0).sum()),
                continuity=hmm.continuity(pa, ctx["arcs"]), route_precision=prec, route_recall=rec, route_f1=f1r,
                sel=sel, point_arcs=pa)


def pooled(trips, method, params=None):
    res = {}
    for tid, d in trips.items():
        rows = hmm.argmax_rows(d["prep"], d["p"]) if method == "classifier" else hmm.decode(d["prep"], d["p"], params)
        res[tid] = evaluate_trip(d, rows)
    return res


def summarise(res, trips, name, split):
    hits = sum(r["hits"] for r in res.values()); n = sum(r["n_pts"] for r in res.values())
    y = np.concatenate([trips[t]["y"] for t in res]); s = np.concatenate([res[t]["sel"] for t in res])
    return dict(split=split, method=name, top1_pooled=hits / n, top1_mean_trip=np.mean([r["top1"] for r in res.values()]),
                macro_f1_selected=f1_score(y, s, average="macro"),
                continuity_mean_trip=np.mean([r["continuity"] for r in res.values()]),
                route_precision=np.mean([r["route_precision"] for r in res.values()]),
                route_recall=np.mean([r["route_recall"] for r in res.values()]),
                route_f1=np.mean([r["route_f1"] for r in res.values()]),
                trips_top1_ge_0_95=int(sum(r["top1"] >= 0.95 for r in res.values())), n_trips=len(res),
                unmatched_points=int(sum(r["unmatched"] for r in res.values())), n_points=n)


# ---------------------------------------------------------------- 3. decoder tuning on train OOF only (Optuna TPE)
def objective(trial):
    q = dict(w_emis=trial.suggest_float("w_emis", 0.3, 3.0, log=True),
             beta=trial.suggest_float("beta", 2.0, 60.0, log=True),
             disc_pen=trial.suggest_float("disc_pen", 4.0, 40.0, log=True),
             uturn_pen=trial.suggest_float("uturn_pen", 0.0, 12.0),
             back_pen=trial.suggest_float("back_pen", 0.0, 8.0),
             gap_m=trial.suggest_categorical("gap_m", [300.0, 600.0, 1000.0, 2000.0]))
    res = pooled(TR, "hmm", q)
    hits = sum(r["hits"] for r in res.values()); n = sum(r["n_pts"] for r in res.values())
    trial.set_user_attr("continuity", float(np.mean([r["continuity"] for r in res.values()])))
    return hits / n


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8))
study.enqueue_trial(hmm.DEFAULT_PARAMS)                                  # trial 0 = hand-picked defaults
t0 = time.time()
study.optimize(objective, n_trials=N_TRIALS,
               callbacks=[lambda s, t: print(f"decoder trial {t.number:>2} train top-1={t.value:.4f} best={s.best_value:.4f} {t.params}", flush=True)])
dec_s = time.time() - t0
study.trials_dataframe().to_csv(data_path("hmm_decoder_trials.csv"), index=False)
TUNED = dict(study.best_params)
json.dump(dict(params=TUNED, defaults=hmm.DEFAULT_PARAMS, objective="pooled train OOF Top-1", best_train_top1=study.best_value,
               n_trials=len(study.trials), search_seconds=round(dec_s, 1), sampler="TPE", seed=SEED,
               back_tol_m=hmm.BACK_TOL, max_limit_m=hmm.MAX_LIMIT),
          open(os.path.join(MODELS, "hmm_config.json"), "w"), indent=2)
print(f"decoder search {dec_s:.0f}s, best {TUNED}")

# ---------------------------------------------------------------- 4. classifier-only vs HMM
rows, per_trip = [], []
for split, trips in [("train_oof", TR), ("student_test", TE)]:
    for name, method, q in [("classifier only (argmax)", "classifier", None), ("classifier + HMM (default params)", "hmm", hmm.DEFAULT_PARAMS),
                            ("classifier + HMM (tuned params)", "hmm", TUNED)]:
        t0 = time.time(); res = pooled(trips, method, q); dt = time.time() - t0
        r = summarise(res, trips, name, split); r["decode_seconds"] = dt; rows.append(r)
        for tid, x in res.items():
            per_trip.append(dict(split=split, method=name, trip=tid, **{k: v for k, v in x.items() if k not in ("sel", "point_arcs")}))
        if split == "student_test" and method == "classifier": CLS = res
        if split == "student_test" and q is TUNED: HMMR = res
R = pd.DataFrame(rows); R.to_csv(data_path("hmm_vs_classifier.csv"), index=False)
pd.DataFrame(per_trip).to_csv(data_path("hmm_per_trip.csv"), index=False)

pred = []
for tid, d in TE.items():
    ref_arc = PL[(PL.split == "student_test") & (PL.trip == tid)].sort_values("gps_idx").lab_arc.values
    pred.append(pd.DataFrame(dict(trip=tid, gps_idx=np.arange(d["ctx"]["n"]), classifier_arc=CLS[tid]["point_arcs"],
                                  hmm_arc=HMMR[tid]["point_arcs"], ref_arc=ref_arc)))
pd.concat(pred).to_csv(data_path("student_test_predictions.csv"), index=False)

json.dump(dict(model="HistGradientBoosting (tuned)", params=BEST, fit_rows=len(fit), fit_seconds=fit_s,
               infer_ms_per_1000_rows=inf_ms, test_logloss=float(log_loss(te.y, np.clip(te.p, 1e-6, 1 - 1e-6))),
               test_macro_f1_at_0_5=float(f1_score(te.y, te.p >= .5, average="macro"))),
          open(data_path("final_model_metrics.json"), "w"), indent=2)
pd.set_option("display.width", 250, "display.max_columns", 30)
print("\n=== CLASSIFIER-ONLY vs HMM ===\n", R.round(4).to_string(index=False))
print(f"total {time.time()-t_all:.0f}s")
