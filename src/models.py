import os, glob, json, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathsnap_core import find_student_root, DATA, MODELS
ROOT = find_student_root()
os.makedirs(MODELS, exist_ok=True); os.chdir(DATA)        # all inputs/outputs live in data/ (run from anywhere)
SEED, N_FIT, N_FOLDS = 42, 300_000, 4
FEATS = json.load(open("feature_cols.json"))

tr = pd.read_pickle("features_train.pkl"); te = pd.read_pickle("features_student_test.pkl")
S = pd.read_csv("candidate_build_summary.csv", dtype={"trip": str}); S["trip"] = S.trip.astype(str).str.zfill(8)
tr["trip"] = tr.trip.astype(str); te["trip"] = te.trip.astype(str)
tr = tr[~tr.suspect].reset_index(drop=True)                  # decision: exclude suspect trips 33, 54 from training/CV
npts = S.set_index(["split", "trip"]).n_gps                  # points per trip (denominator for top-1, incl. points with no candidate)
print(f"Training rows (non-suspect): {len(tr):,} | trips: {tr.trip.nunique()} | student-test rows: {len(te):,}")

def Xmat(df): return np.nan_to_num(df[FEATS].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def sample_points(df, n_rows, seed):                         # sample whole GPS points so candidate groups stay intact
    if len(df) <= n_rows: return df
    g = df.groupby(["trip", "gi"], sort=False).ngroup().values
    rng = np.random.default_rng(seed); frac = n_rows / len(df)
    keep = rng.random(g.max() + 1) < frac
    return df[keep[g]]

def top1(df, p, split):                                      # observation-level Top-1 accuracy
    d = df[["trip", "gi", "y", "arc"]].copy(); d["p"] = p
    best = d.loc[d.groupby(["trip", "gi"]).p.idxmax()]
    hits = best.groupby("trip").y.sum()
    n = sum(npts[(split, t)] for t in hits.index)
    return float(hits.sum() / n), best

def load_arcs(split, tid):
    return pd.read_csv(f"{ROOT}/{split}/{tid}/{tid}.arcs", sep=r"\s+", header=None).values.astype(np.int64)

def continuity(best, split):                                 # fraction of consecutive predicted arcs that are same/connected
    ok = tot = 0
    for tid, g in best.groupby("trip"):
        ar = load_arcs(split, tid); g = g.sort_values("gi"); a = g.arc.values
        if len(a) < 2: continue
        same = (ar[a[:-1]] == ar[a[1:]]).all(1); conn = ar[a[:-1], 1] == ar[a[1:], 0]
        ok += int((same | conn).sum()); tot += len(a) - 1
    return ok / max(tot, 1)

MODELS = {
 "LogReg": ("Linear", dict(C=1.0), lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300, random_state=SEED))),
 "GaussianNB": ("Probabilistic", dict(var_smoothing=1e-9), lambda: GaussianNB()),
 "DecisionTree": ("Tree", dict(max_depth=12, min_samples_leaf=20), lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED)),
 "RandomForest": ("Ensemble (bagging)", dict(n_estimators=100, max_depth=14, min_samples_leaf=5), lambda: RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
 "HistGB": ("Boosting", dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31), lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=SEED)),
 "MLP": ("Neural net", dict(hidden=(64, 32), max_iter=30), lambda: make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=30, early_stopping=True, random_state=SEED))),
}
try:
    import lightgbm as lgb
    MODELS["LightGBM"] = ("Boosting", dict(n_estimators=300, learning_rate=0.1, num_leaves=31),
                          lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31, random_state=SEED, verbose=-1, n_jobs=-1))
except Exception as e:
    print("LightGBM unavailable, skipping:", str(e)[:80])

# reference ceiling: continuity of the aligned reference labels on student-test (what a perfect classifier would score)
pl = pd.read_csv("point_labels.csv", dtype={"trip": str}); pl["trip"] = pl.trip.str.zfill(8)
ref = pl[pl.split == "student_test"].rename(columns={"gps_idx": "gi", "lab_arc": "arc"})
ref_cont = continuity(ref, "student_test"); print(f"Reference-label route continuity ceiling (student-test): {ref_cont:.4f}")

gkf = GroupKFold(n_splits=N_FOLDS); groups = tr.trip.values
rows, fold_rows, t_all = [], [], time.time()
for name, (family, cfg, make) in MODELS.items():
    t0 = time.time(); cvf1, cvtop, cvll, trf1 = [], [], [], []
    for k, (a, b) in enumerate(gkf.split(tr, groups=groups)):
        fit = sample_points(tr.iloc[a], N_FIT, SEED + k); val = tr.iloc[b]
        m = make().fit(Xmat(fit), fit.y.values)
        pv = m.predict_proba(Xmat(val))[:, 1]; ptr = m.predict_proba(Xmat(fit))[:, 1]
        t1 = np.mean([top1(val[val.trip == t], pv[(val.trip == t).values], "train")[0] for t in val.trip.unique()])
        cvf1.append(f1_score(val.y, pv >= .5, average="macro")); cvtop.append(t1)
        cvll.append(log_loss(val.y, np.clip(pv, 1e-6, 1 - 1e-6))); trf1.append(f1_score(fit.y, ptr >= .5, average="macro"))
        fold_rows.append(dict(model=name, fold=k, macro_f1=cvf1[-1], top1=t1, log_loss=cvll[-1]))
    # refit on a sample of ALL non-suspect train trips -> student-test
    fit = sample_points(tr, N_FIT, SEED); t_fit = time.time()
    m = make().fit(Xmat(fit), fit.y.values); train_time = time.time() - t_fit
    t_inf = time.time(); pt = m.predict_proba(Xmat(te))[:, 1]; inf_ms = (time.time() - t_inf) / len(te) * 1000 * 1000
    s_top, best = top1(te, pt, "student_test")
    rows.append(dict(model=name, family=family, config=json.dumps(cfg),
        cv_macro_f1=np.mean(cvf1), cv_macro_f1_std=np.std(cvf1), cv_top1=np.mean(cvtop), cv_top1_std=np.std(cvtop), cv_logloss=np.mean(cvll),
        train_macro_f1=np.mean(trf1), test_top1=s_top, test_macro_f1=f1_score(te.y, pt >= .5, average="macro"),
        test_logloss=log_loss(te.y, np.clip(pt, 1e-6, 1 - 1e-6)), test_route_continuity=continuity(best, "student_test"),
        fit_seconds=train_time, infer_ms_per_1000_rows=inf_ms))
    r = rows[-1]; print(f"{name:13s} CV top1={r['cv_top1']:.4f}±{r['cv_top1_std']:.4f} CV F1={r['cv_macro_f1']:.4f} | "
        f"test top1={r['test_top1']:.4f} F1={r['test_macro_f1']:.4f} LL={r['test_logloss']:.4f} cont={r['test_route_continuity']:.4f} "
        f"| train F1={r['train_macro_f1']:.4f} | fit {train_time:.0f}s | {time.time()-t0:.0f}s total")

R = pd.DataFrame(rows).sort_values("cv_top1", ascending=False)
R.to_csv("model_comparison.csv", index=False); pd.DataFrame(fold_rows).to_csv("cv_fold_scores.csv", index=False)
json.dump(dict(seed=SEED, n_fit_rows=N_FIT, n_folds=N_FOLDS, features=FEATS, excluded_trips=sorted(set(S[S.suspect].trip)) if "suspect" in S else []),
          open("experiment_config.json", "w"), indent=2)
print("\n=== MODEL COMPARISON (sorted by CV top-1) ===")
pd.set_option("display.width", 250, "display.max_columns", 30)
print(R.drop(columns=["config"]).round(4).to_string(index=False))
print(f"\nReference continuity ceiling: {ref_cont:.4f} | heuristic baseline (nearest+heading) student-test top-1: ~0.850")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 4)); x = np.arange(len(R))
ax[0].bar(x, R.cv_top1, yerr=R.cv_top1_std); ax[0].set_title("CV Top-1 (group CV, train trips)")
ax[1].bar(x - .2, R.test_top1, .4, label="Top-1"); ax[1].bar(x + .2, R.test_macro_f1, .4, label="Macro F1"); ax[1].set_title("Student-test"); ax[1].legend()
ax[2].bar(x, R.test_logloss); ax[2].set_title("Student-test log loss")
for a in ax: a.set_xticks(x); a.set_xticklabels(R.model, rotation=45, ha="right")
plt.tight_layout(); plt.savefig("model_comparison.png", dpi=150); print(f"Saved model_comparison.csv / .png | total {time.time()-t_all:.0f}s")