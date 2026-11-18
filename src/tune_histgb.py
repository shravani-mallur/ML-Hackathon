import json, time, warnings
import numpy as np, pandas as pd, optuna, joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathsnap_core import find_student_root, DATA, MODELS
ROOT = find_student_root()
os.makedirs(MODELS, exist_ok=True); os.chdir(DATA)        # all inputs/outputs live in data/ (run from anywhere)
SEED, N_FIT, N_FOLDS, N_TRIALS, TIMEOUT_S = 42, 300_000, 4, 40, 1800
FEATS = json.load(open("feature_cols.json"))
tr = pd.read_pickle("features_train.pkl"); te = pd.read_pickle("features_student_test.pkl")
S = pd.read_csv("candidate_build_summary.csv", dtype={"trip": str}); S["trip"] = S.trip.astype(str).str.zfill(8)
tr["trip"] = tr.trip.astype(str); te["trip"] = te.trip.astype(str)
tr = tr[~tr.suspect].reset_index(drop=True)                       # same exclusions as the model comparison
npts = S.set_index(["split", "trip"]).n_gps

def Xmat(df): return np.nan_to_num(df[FEATS].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def sample_points(df, n_rows, seed):                              # same as before: whole GPS points
    if len(df) <= n_rows: return df
    g = df.groupby(["trip", "gi"], sort=False).ngroup().values
    keep = np.random.default_rng(seed).random(g.max() + 1) < n_rows / len(df)
    return df[keep[g]]

def best_rows(df, p):
    d = df[["trip", "gi", "y"]].copy(); d["p"] = p
    return d.loc[d.groupby(["trip", "gi"]).p.idxmax()]

def top1_by_trip(df, p, split):                                   # CV metric (mean over trips), as in models.py
    hits = best_rows(df, p).groupby("trip").y.sum()
    return float(np.mean([hits[t] / npts[(split, t)] for t in hits.index]))

def top1_pooled(df, p, split):                                    # student-test metric (pooled over points)
    hits = best_rows(df, p).groupby("trip").y.sum()
    return float(hits.sum() / sum(npts[(split, t)] for t in hits.index))

# identical folds and samples to the comparison script
folds = []
for k, (a, b) in enumerate(GroupKFold(n_splits=N_FOLDS).split(tr, groups=tr.trip.values)):
    fit = sample_points(tr.iloc[a], N_FIT, SEED + k); val = tr.iloc[b]
    folds.append((Xmat(fit), fit.y.values, Xmat(val), val))

def cv_eval(params):
    tops, f1s, lls = [], [], []
    for Xf, yf, Xv, val in folds:
        pv = HistGradientBoostingClassifier(random_state=SEED, **params).fit(Xf, yf).predict_proba(Xv)[:, 1]
        tops.append(top1_by_trip(val, pv, "train"))
        f1s.append(f1_score(val.y, pv >= .5, average="macro")); lls.append(log_loss(val.y, np.clip(pv, 1e-6, 1 - 1e-6)))
    return float(np.mean(tops)), float(np.mean(f1s)), float(np.mean(lls)), tops

def objective(trial):
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        max_iter=trial.suggest_int("max_iter", 100, 500, step=50),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 8, 127, log=True),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 300, log=True),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        max_bins=trial.suggest_categorical("max_bins", [63, 127, 255]))
    top, f1, ll, _ = cv_eval(params)
    trial.set_user_attr("cv_macro_f1", f1); trial.set_user_attr("cv_logloss", ll)
    return top                                                    # objective: mean group-CV observation-level Top-1

UNTUNED = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31)   # identical to the comparison run
def progress(study, trial):
    secs = (trial.datetime_complete - trial.datetime_start).total_seconds()
    print(f"trial {trial.number:>2} | cv top-1={trial.value:.4f} | best so far={study.best_value:.4f} | {secs:.0f}s | {trial.params}", flush=True)

study = optuna.create_study(direction="maximize", study_name="histgb_top1",
                            storage="sqlite:///optuna_histgb.db", load_if_exists=True,   # saved to disk, resumable
                            sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10))
done = sum(t.state.name == "COMPLETE" for t in study.trials)
t0 = time.time()
try:
    study.optimize(objective, n_trials=max(N_TRIALS - done, 0), timeout=TIMEOUT_S, callbacks=[progress])
except KeyboardInterrupt:
    print("Interrupted: continuing with the trials completed so far.")
search_s = time.time() - t0
study.trials_dataframe().to_csv("optuna_histgb_trials.csv", index=False)
json.dump(dict(best_params=study.best_params, best_cv_top1=study.best_value, n_trials=len(study.trials),
               search_seconds=round(search_s, 1), sampler="TPE", seed=SEED, n_fit_rows=N_FIT, n_folds=N_FOLDS,
               space=dict(learning_rate="log 0.02-0.3", max_iter="100-500 step 50", max_leaf_nodes="log int 8-127",
                          min_samples_leaf="log int 10-300", l2_regularization="log 1e-3-10", max_bins=[63, 127, 255])),
          open("optuna_histgb_best.json", "w"), indent=2)
print(f"\nSearch: {len(study.trials)} trials in {search_s:.0f}s | best CV top-1 = {study.best_value:.4f}\nBest params: {study.best_params}")

def test_eval(params, n_rows, tag, cv=None):
    fit = sample_points(tr, n_rows, SEED); t = time.time()
    m = HistGradientBoostingClassifier(random_state=SEED, **params).fit(Xmat(fit), fit.y.values); fit_s = time.time() - t
    t = time.time(); p = m.predict_proba(Xmat(te))[:, 1]; inf_ms = (time.time() - t) / len(te) * 1e6
    row = dict(model=tag, fit_rows=len(fit), cv_top1=cv[0] if cv else np.nan, cv_macro_f1=cv[1] if cv else np.nan,
               cv_logloss=cv[2] if cv else np.nan, test_top1=top1_pooled(te, p, "student_test"),
               test_macro_f1=f1_score(te.y, p >= .5, average="macro"), test_logloss=log_loss(te.y, np.clip(p, 1e-6, 1 - 1e-6)),
               fit_seconds=fit_s, infer_ms_per_1000_rows=inf_ms)
    return row, m

best = study.best_params
cv_un, cv_tu = cv_eval(UNTUNED), cv_eval(best)
r1, m1 = test_eval(UNTUNED, N_FIT, "HistGB untuned", cv_un)
r2, m2 = test_eval(best, N_FIT, "HistGB tuned (same 300k protocol)", cv_tu)
r3, m3 = test_eval(best, 1_000_000, "HistGB tuned (1M rows, extra)")
R = pd.DataFrame([r1, r2, r3]); R.to_csv("tuned_vs_untuned.csv", index=False)
pd.DataFrame(dict(fold=range(N_FOLDS), untuned_top1=cv_un[3], tuned_top1=cv_tu[3])).to_csv("tuned_vs_untuned_folds.csv", index=False)
joblib.dump(m2, os.path.join(MODELS, "histgb_tuned_300k.joblib")); joblib.dump(m3, os.path.join(MODELS, "histgb_tuned_1M.joblib"))
pd.set_option("display.width", 250, "display.max_columns", 30)
print("\n=== TUNED VS UNTUNED ===\n", R.round(4).to_string(index=False))
print("\nPer-fold CV top-1:\n", pd.read_csv("tuned_vs_untuned_folds.csv").round(4).to_string(index=False))
print("\nTop 5 trials:\n", study.trials_dataframe().sort_values("value", ascending=False).head(5)[
      ["number", "value", "params_learning_rate", "params_max_leaf_nodes", "params_min_samples_leaf", "params_l2_regularization"]].round(4).to_string(index=False))