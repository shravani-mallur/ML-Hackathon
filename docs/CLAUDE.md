# CLAUDE.md: PathSnap (Distributed Map-Matching & Route Reconstruction)

PES University, UE24CS352A Machine Learning hackathon. Team of 4, beginners, 5-hour time limit.
Goal: reconstruct a vehicle's route from noisy 1 Hz GPS by matching it to a road network, using
candidate classification + HMM/Viterbi, served through FastAPI (Laptop 1) and a UI (Laptop 2).

## Non-negotiable rules (from the organisers' documents)

- **Evaluator-test data (10 trips) is never touched during development.** It is not in the repo folder.
  Never use it for training, features, thresholds, model or hyperparameter choice.
- **Ground-truth labels stay server-side.** `.route` data must never reach the prediction model or the UI.
- **No hardcoded predictions, no bypassing the API.** API logs must be shown during final evaluation.
- **Split by trip, never by row.** Cross-validation is always `GroupKFold` with trip as the group.
- Fit all learned preprocessing on the 80 training trips only.
- Use the 10 student-test trips only for development-stage comparisons and decoder tuning, never the evaluator trips.
- Every number in the report must be reproducible from this repo. **Never invent or estimate results.**
- Minimum 5 models compared, Bayesian hyperparameter optimisation with the search record kept,
  HMM vs classifier-only comparison, Folium maps, working API + UI for all 10 evaluator trips.
- AI assistance must be disclosed. Every team member must be able to explain the code, experiments,
  HMM/Viterbi and Bayesian optimisation in the viva. When writing code, explain each step briefly.
- No secrets or credentials in the repo. Report is one consolidated PDF; repo needs a README.

## Data

Location: `Problem-Statement/Student folder/Student folder/{train, student_test}/<trip_id>/<trip_id>.{track,nodes,arcs,route}`
(80 train trips, 10 student_test trips, all with all four files). All files are whitespace-separated, no headers.

| File | Columns |
|---|---|
| `.track` | lon, lat, timestamp (s), nominally 1 Hz |
| `.nodes` | lon, lat (row index = node id) |
| `.arcs` | from_node, to_node (row index = arc_id) |
| `.route` | ordered reference arc_ids (one per route step, not one per GPS point) |

Verified facts (from our EDA):
- Every trip has its **own** road network (90 distinct node/arc counts); arc ids are local to a trip.
  So this is NOT a "predict which arc id" classifier. It is **candidate scoring**: each row is a (GPS point, candidate arc) pair.
- Coordinate order is lon, lat. Trips are in different countries; project each trip to local metres
  (equirectangular, centred on the trip's mean lon/lat).
- Network covers roughly a 1 km buffer around the trip. Arcs are straight node-to-node segments; GPS can be far
  from nodes, so use point-to-segment distance.
- About 92% of arcs have a reverse-direction twin and duplicate arcs exist (1,903 total); arcs with the same
  (from, to) pair are treated as the same label. Direction (heading) is what separates twins.
- Time gaps are not distance gaps (e.g. 153,064 s gap with a 3.3 m jump), so the HMM transition must use spatial distance, not elapsed time.
- Some trips have >50 m/s steps (trips 1, 13, 32 most) that are not yet explained; inspect before drawing conclusions.
- Reference route chain is not perfectly connected in 11 of 90 trips (continuity 98.1-99.9%). The reference is treated as reference, not perfect truth.
- **Suspect trips 00000033 and 00000054 (train):** reference route sits 120-140 m from GPS. Decision: excluded from model
  training and CV, documented as a reference-route limitation. Trips 00000072 and 00000002 also show elevated candidate misses
  (about 10-11%) with a normal median distance; mention them in the data-quality log.
- Per-point labels are **derived** by a monotone dynamic-programming alignment of GPS points to the ordered route
  (assumption; document it and be ready to explain it in the viva). Saved in `point_labels.csv`.

## Pipeline and scripts (run in this order)

All scripts live in `src/`, run from the repo root, and read/write `data/` (results), `models/`, `figures/`, `maps/`.
Shared code: `pathsnap_core.py` (geometry, inference-only `build_candidates`, training-time `label_points`).

1. `data_processing.py` (replaces the missing `input_proccessing.py`): inventory, relationship and connectivity checks. Writes `dataset_inventory.csv`, `network_connectivity.csv`.
2. `folium_review.py`: derives labels, candidate miss rate by radius and top-K, trip-level checks, Folium map.
   Writes `trip_stats.csv`, `candidate_sample.csv`, `eda_map_train_00000046.html`.
3. `feature.py`: builds candidates and features for all trips. Writes `features_train.pkl`, `features_student_test.pkl`,
   `feature_cols.json`, `candidate_build_summary.csv`, `point_labels.csv`.
4. `models.py`: 7-model comparison with 4-fold GroupKFold + student-test evaluation.
   Writes `model_comparison.csv`, `cv_fold_scores.csv`, `experiment_config.json`, `model_comparison.png`.
5. Bayesian optimisation of HistGradientBoosting (Optuna TPE). Writes `optuna_histgb_trials.csv`, `optuna_histgb_best.json`,
   `tuned_vs_untuned.csv`, `tuned_vs_untuned_folds.csv`, `histgb_tuned_*.joblib`. Script: `tune_histgb.py`.
6. `hmm_experiment.py` (+ `hmm.py`): OOF emissions on train, final model `models/histgb_tuned.joblib`, decoder tuned with Optuna on train OOF
   (`models/hmm_config.json`), classifier-only vs HMM (`hmm_vs_classifier.csv`, `hmm_per_trip.csv`). Student-test Top-1 0.8926 -> 0.9456, continuity 0.940 -> 0.978.
7. `make_maps.py` (Folium, via `route_map.py`), `plots.py` (figures). `api/main.py` (FastAPI), `ui/app.py` (UI), `README.md`, `docs/report.md` (draft).
   Evaluator trips live only in git-ignored `server_data/evaluator_test/` on Laptop 1.

## Key design decisions

- **Candidate rule:** nearest 15 arcs within 150 m of each GPS point (point-to-segment distance). Student-test candidate miss rate: 0.09%.
- **Features (19):** `d, rank, d_rel, t, L, cos_h, side_d, dend, d_prev, d_next, twin, indeg_u, outdeg_v, n_cand, speed, dt_prev, step_prev, turn_cos, hvalid`
  (see `feature_cols.json`). Heading (`cos_h`) is essential because of twin arcs.
- **Label:** candidate row is positive if its (from, to) key equals the aligned reference arc's key. Class balance about 7.6% positive.
- **Primary selection metric:** observation-level Top-1 accuracy under group CV (per point, pick the highest-scoring candidate; points with no candidate count as wrong). Route continuity is the tie-breaker. Log loss matters because the HMM consumes probabilities.
- **Route continuity definition:** fraction of consecutive predicted arcs that are the same arc or where arc i's end node equals arc i+1's start node.
  Reference-label ceiling on student-test: **0.9851**.
- **Comparison protocol:** seed 42, 4-fold GroupKFold on the 78 non-suspect train trips, each fit uses a 300k-row sample of whole GPS points (candidate groups kept intact).
  CV Top-1 is averaged per trip; student-test Top-1 is pooled over points. Do not compare them across columns.
- **Chosen model for tuning:** HistGradientBoosting (tied with RandomForest on CV Top-1, about 8x cheaper to fit, smaller train/CV gap).
  This choice is open to challenge and must be justified with `model_comparison.csv` and `cv_fold_scores.csv`.

## Results so far (student-test = 10 dev trips; all from `model_comparison.csv`)

Hand heuristic (nearest arc + heading penalty): student-test Top-1 about 0.850. Nearest arc only: 0.538.

| Model | Family | CV Top-1 | CV Macro F1 | Test Top-1 | Test Macro F1 | Test log loss | Test continuity | Fit s |
|---|---|---|---|---|---|---|---|---|
| RandomForest | Ensemble | 0.8848 | 0.9327 | 0.8805 | 0.9321 | 0.0525 | 0.9288 | 10.9 |
| HistGB | Boosting | 0.8845 | 0.9316 | 0.8769 | 0.9298 | 0.0557 | 0.9260 | 1.3 |
| MLP | Neural | 0.8798 | 0.9257 | 0.8852 | 0.9304 | 0.0548 | 0.9359 | 17.6 |
| DecisionTree | Tree | 0.8794 | 0.9261 | 0.8664 | 0.9236 | 0.0788 | 0.9231 | 4.9 |
| LightGBM | Boosting | 0.8762 | 0.9291 | 0.8849 | 0.9333 | 0.0559 | 0.9255 | 1.5 |
| LogReg | Linear | 0.8678 | 0.9122 | 0.8546 | 0.9092 | 0.0774 | 0.9208 | 0.7 |
| GaussianNB | Probabilistic | 0.8378 | 0.7548 | 0.8127 | 0.7485 | 0.6457 | 0.9127 | 0.1 |

Paired fold analysis vs HistGB: DecisionTree, LightGBM, LogReg, GaussianNB lose on CV Top-1 in all 4 folds; MLP loses in 3 of 4;
RandomForest is a statistical tie. Differences under about 1 pt on 10 test trips are noise. Models used default settings and only
about 13% of the training rows, so the ranking is conditional on that. Classifier-only continuity (0.913-0.936) is well below the 0.985 ceiling; the HMM should close that gap.

## HMM / Viterbi design (implemented in `src/hmm.py`; this was the plan)

- Hidden states: candidate arcs per GPS point (preserve candidate-to-arc correspondence).
- Emission: the tuned classifier's calibrated probability (or its log) for each candidate.
- Transition: depends on graph connectivity (same arc, connected arc, or reachable within a small number of hops) and on the spatial distance between consecutive GPS points, not elapsed time; penalise disconnected and reversing transitions.
- Handle: points with no candidates, large GPS gaps (reset or relax transitions), invalid transitions.
- Tune decoder parameters on train + student-test only. Report classifier-only vs classifier+HMM pointwise accuracy, sequence-level performance and route continuity, including cases where the HMM does not help.

## API / UI plan

- Evaluator trips ship as raw files only (`.track`, `.nodes`, `.arcs`, plus `.route` kept server-side for scoring).
  Split `feature.py` into an **inference-only feature builder** (GPS + network only, no `.route`) and a training-time labeller.
- API (Laptop 1): `trip_id` in; per-point predicted arc id and coordinates, predicted route geometry, and server-computed accuracy out. Log every request.
- UI (Laptop 2): lists the 10 evaluation trips, sends the selected trip id to the API, shows the Folium map (raw GPS, road network, reference route, predicted route, layer control, legend) and metrics.
- Report must include screenshots of the UI and of API responses/logs on the receiving laptop.

## Conventions for Claude working in this repo

- Complete, runnable Python (pandas, numpy, scikit-learn, scipy, lightgbm, optuna, folium, FastAPI). Python 3.14 is in use on Windows; if a package lacks wheels, use a 3.12 virtual environment.
- Fixed seed 42 everywhere; log configurations and results to CSV/JSON in the repo.
- Never invent column names, file names or numbers; ask for `df.head()`, `df.info()` or output when unsure.
- If something looks like leakage, a wrong metric or a wrong split, say so and fix it.
- Prefer the simple option that works; add complexity only if it measurably improves group-CV Top-1.
- Do not draw training curves for models without epoch histories.
- Time boxes: submit a working end-to-end pipeline first, then improve. Reserve the final stretch for API/UI dry runs on two laptops, the report and the README.

## Open items

- Open trips 00000033 and 00000054 in Folium and screenshot the route/GPS mismatch for the data-quality section.
- Investigate the >50 m/s steps (trips 1, 13, 32) before the report.
- Done: tuning, HMM/Viterbi, inference-only features, FastAPI, UI, Folium maps, README, report draft.
- Report draft (`docs/report.md`): team to rewrite explanations, fill per-model TODOs, add real two-laptop screenshots, export one PDF.
- UI does not show the reference route (rule: `.route` must not reach the UI); confirm with organisers if they want it shown.
- Confirm with organisers how per-point reference labels and "route continuity" are defined, in case ours differ.