"""Report figures, all generated from saved experiment outputs (no numbers typed by hand).

EDA        : missing values / anomalies per trip, sampling-interval distribution, GPS trajectory + road network plots
Models     : CV Macro F1, CV Top-1 per fold, student-test Top-1 + Macro F1, log loss, MLP loss curve (only model
             with an epoch history), confusion matrices of the tuned model
Tuning     : Optuna optimisation history, tuned vs untuned
HMM        : classifier-only vs classifier+HMM (pooled and per trip)
Outputs: figures/*.png
Run from the repository root:  python src/plots.py
"""
import os, sys, json, warnings
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathsnap_core import FIGS, MODELS, find_student_root, data_path, load_trip_inputs, load_route, trip_base, project, feature_matrix
warnings.filterwarnings("ignore")
os.makedirs(FIGS, exist_ok=True); ROOT = find_student_root(); SEED = 42


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, name), dpi=150); plt.close(fig); print("saved figures/" + name)


def zoom(ax, vals, pad=0.01):
    lo, hi = np.nanmin(vals), np.nanmax(vals); ax.set_ylim(max(0, lo - pad), min(1, hi + pad) if hi <= 1 else hi + pad)


# ------------------------------------------------------------------ EDA (train + student-test partitions)
inv = pd.read_csv(data_path("dataset_inventory.csv")); ts = pd.read_csv(data_path("trip_stats.csv"), dtype={"trip": str})
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
miss = inv.groupby("split")[["nan", "dup_ts", "neg_ts", "self_loops"]].sum()
miss.plot.bar(ax=ax[0], rot=0); ax[0].set_title("Missing / invalid values per partition (counts)"); ax[0].set_ylabel("count")
ax[1].hist(inv.pct_1s, bins=30, color="#1f6fd1"); ax[1].set_title("% of 1-second sampling steps per trip"); ax[1].set_xlabel("% steps = 1 s")
ax[2].hist(np.log10(inv.max_gap_s.clip(lower=1)), bins=30, color="#d7301f"); ax[2].set_title("Largest time gap per trip"); ax[2].set_xlabel("log10(max gap, s)")
save(fig, "eda_missing_and_sampling.png")

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].hist(inv.n_gps, bins=30, color="#1f6fd1"); ax[0].set_title("GPS observations per trip"); ax[0].set_xlabel("points")
ax[1].scatter(inv.n_nodes, inv.n_arcs, s=12, c=(inv.split == "train").map({True: "#1f6fd1", False: "#d7301f"}))
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("nodes"); ax[1].set_ylabel("arcs"); ax[1].set_title("Road-network size per trip (blue train, red student-test)")
ax[2].hist(ts.d_true_median, bins=40, color="#1a9850"); ax[2].set_yscale("log"); ax[2].set_title("Median GPS-to-reference distance per trip"); ax[2].set_xlabel("metres")
save(fig, "eda_trip_sizes.png")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, (split, tid) in zip(axes, [("train", "00000046"), ("train", "00000033"), ("student_test", "00000009")]):
    base = trip_base(f"{ROOT}/{split}", tid); tk, nd, ar = load_trip_inputs(base); rt = load_route(base)
    lon0, lat0 = tk[:, 0].mean(), tk[:, 1].mean(); P, N = project(tk[:, :2], lon0, lat0), project(nd, lon0, lat0)
    lim = np.abs(P).max() + 300; keep = (np.abs(N[ar[:, 0]]) < lim).all(1)
    for a in np.where(keep)[0][:20000]:
        ax.plot(N[ar[a], 0], N[ar[a], 1], color="#bbbbbb", lw=0.5)
    for a in rt: ax.plot(N[ar[a], 0], N[ar[a], 1], color="#1a9850", lw=3, alpha=0.7)
    ax.scatter(P[:, 0], P[:, 1], s=2, color="#1f6fd1", zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title(f"{split} {tid}: GPS (blue), network (grey), reference (green)", fontsize=10); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
save(fig, "eda_trajectories_network.png")

# ------------------------------------------------------------------ model comparison
R = pd.read_csv(data_path("model_comparison.csv")); F = pd.read_csv(data_path("cv_fold_scores.csv"))
x = np.arange(len(R))
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x, R.cv_macro_f1, yerr=R.cv_macro_f1_std, color="#1f6fd1", capsize=4)
ax.set_xticks(x); ax.set_xticklabels(R.model, rotation=30, ha="right"); ax.set_ylim(0.7, 0.96)
ax.set_title("4-fold GroupKFold CV Macro F1 (78 train trips), mean ± std"); [ax.text(i, v + .003, f"{v:.3f}", ha="center", fontsize=8) for i, v in enumerate(R.cv_macro_f1)]
save(fig, "models_cv_macro_f1.png")

fig, ax = plt.subplots(figsize=(9, 4))
for mdl, g in F.groupby("model"): ax.plot(g.fold, g.top1, marker="o", label=mdl)
ax.set_xlabel("fold"); ax.set_xticks(range(F.fold.max() + 1)); ax.set_ylabel("Top-1 (mean over validation trips)")
ax.set_title("CV observation-level Top-1 per fold"); ax.legend(fontsize=8, ncol=2)
save(fig, "models_cv_top1_folds.png")

fig, ax = plt.subplots(1, 2, figsize=(15, 4))
ax[0].bar(x - .2, R.test_top1, .4, label="Top-1 accuracy"); ax[0].bar(x + .2, R.test_macro_f1, .4, label="Macro F1")
zoom(ax[0], np.r_[R.test_top1, R.test_macro_f1], 0.03); ax[0].set_title("Student-test (10 trips): Top-1 and Macro F1"); ax[0].legend()
ax[1].bar(x, R.test_logloss, color="#d7301f"); ax[1].set_yscale("log"); ax[1].set_title("Student-test log loss (log scale, lower is better)")
for a in ax: a.set_xticks(x); a.set_xticklabels(R.model, rotation=30, ha="right")
save(fig, "models_student_test.png")

# MLP: the only compared model with an epoch-wise history -> refit with the exact comparison configuration
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
tr = pd.read_pickle(data_path("features_train.pkl")); tr = tr[~tr.suspect].reset_index(drop=True)
g = tr.groupby(["trip", "gi"], sort=False).ngroup().values
fit = tr[(np.random.default_rng(SEED).random(g.max() + 1) < 300_000 / len(tr))[g]]
sc = StandardScaler().fit(feature_matrix(fit))
mlp = MLPClassifier((64, 32), max_iter=30, early_stopping=True, random_state=SEED).fit(sc.transform(feature_matrix(fit)), fit.y.values)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(np.arange(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_, marker="o"); ax[0].set_title("MLP training loss (log loss) per epoch"); ax[0].set_xlabel("epoch")
ax[1].plot(np.arange(1, len(mlp.validation_scores_) + 1), mlp.validation_scores_, marker="o", color="#d7301f")
ax[1].set_title("MLP internal validation accuracy (10% early-stopping split)"); ax[1].set_xlabel("epoch")
save(fig, "mlp_training_curve.png")
del tr, fit

# ------------------------------------------------------------------ Bayesian optimisation
T = pd.read_csv(data_path("optuna_histgb_trials.csv")); T = T[T.state == "COMPLETE"]
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.scatter(T.number, T.value, s=18, label="trial CV Top-1"); ax.plot(T.number, T.value.cummax(), color="#d7301f", label="best so far")
ax.set_xlabel("trial"); ax.set_ylabel("4-fold group-CV Top-1"); ax.set_title("Optuna TPE search for HistGradientBoosting"); ax.legend()
save(fig, "optuna_history.png")

U = pd.read_csv(data_path("tuned_vs_untuned.csv"))
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8)); lab = ["untuned", "tuned (300k)", "tuned (1M)"]
for a, cols, t in [(ax[0], ["cv_top1", "cv_macro_f1"], "Cross-validation"), (ax[1], ["test_top1", "test_macro_f1"], "Student-test")]:
    xx = np.arange(len(U))
    for k, c in enumerate(cols): a.bar(xx + (k - .5) * .4, U[c], .4, label=c)
    zoom(a, U[cols].values.ravel(), 0.01); a.set_xticks(xx); a.set_xticklabels(lab); a.set_title(t); a.legend(fontsize=8)
ax[2].bar(lab, U.infer_ms_per_1000_rows, color="#888"); ax[2].set_title("Inference time (ms / 1000 candidate rows)")
save(fig, "tuned_vs_untuned.png")

# ------------------------------------------------------------------ tuned model confusion matrices (student-test)
te = pd.read_pickle(data_path("features_student_test.pkl"))
model = joblib.load(os.path.join(MODELS, "histgb_tuned.joblib")); p = model.predict_proba(feature_matrix(te))[:, 1]
te["p"] = p; sel = np.zeros(len(te), int); sel[te.groupby(["trip", "gi"]).p.idxmax().values] = 1
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(confusion_matrix(te.y, p >= .5), display_labels=["not ref", "ref arc"]).plot(ax=ax[0], values_format="d", colorbar=False)
ax[0].set_title("Tuned HistGB, threshold 0.5 (candidate rows)")
ConfusionMatrixDisplay(confusion_matrix(te.y, sel), display_labels=["not ref", "ref arc"]).plot(ax=ax[1], values_format="d", colorbar=False)
ax[1].set_title("Tuned HistGB, argmax per GPS point")
save(fig, "confusion_tuned_histgb.png")

# ------------------------------------------------------------------ HMM vs classifier
H = pd.read_csv(data_path("hmm_vs_classifier.csv")); PT = pd.read_csv(data_path("hmm_per_trip.csv"), dtype={"trip": str})
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
for a, split in zip(ax, ["train_oof", "student_test"]):
    h = H[H.split == split]; mets = ["top1_pooled", "macro_f1_selected", "continuity_mean_trip", "route_f1"]; xx = np.arange(len(mets))
    for k, (_, r) in enumerate(h.iterrows()): a.bar(xx + (k - 1) * .27, [r[m] for m in mets], .27, label=r.method)
    zoom(a, h[mets].values.ravel(), 0.02); a.set_xticks(xx); a.set_xticklabels(["Top-1", "Macro F1 (selected)", "Continuity", "Route F1"])
    a.set_title({"train_oof": "Train (78 trips, out-of-fold emissions)", "student_test": "Student-test (10 trips)"}[split]); a.legend(fontsize=8)
save(fig, "hmm_vs_classifier.png")

st = PT[PT.split == "student_test"].pivot(index="trip", columns="method", values="top1")
fig, ax = plt.subplots(figsize=(11, 4)); xx = np.arange(len(st))
for k, c in enumerate(st.columns): ax.bar(xx + (k - 1) * .27, st[c], .27, label=c)
ax.set_xticks(xx); ax.set_xticklabels(st.index, rotation=30); zoom(ax, st.values.ravel(), 0.03)
ax.set_title("Per-trip Top-1 on student-test: classifier-only vs HMM"); ax.legend(fontsize=8)
save(fig, "hmm_per_trip_student_test.png")
