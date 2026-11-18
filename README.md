# PathSnap: map-matching and route reconstruction

PES University UE24CS352A Machine Learning hackathon.
PathSnap reconstructs a vehicle's route from noisy 1 Hz GPS by matching each GPS point to a road segment (arc):

1. **Candidate generation**: for each GPS point, the nearest 15 arcs within 150 m (point-to-segment distance in a local metric projection).
2. **Candidate classifier**: a tuned HistGradientBoosting model scores every (GPS point, candidate arc) pair with 19 geometric, heading and context features.
3. **HMM + Viterbi**: hidden states are the candidate arcs, emissions are the classifier probabilities, transitions compare road-network distance with GPS distance. Viterbi decoding gives one connected, plausible route.
4. **Deployment**: a FastAPI backend (Laptop 1) runs the whole pipeline and scores it on the server. A web UI (Laptop 2) lists the 10 evaluation trips, sends the chosen trip id to the API and shows the route on a Folium map with the metrics.

The technical report is in [`docs/report.md`](docs/report.md). Every number in it comes from the CSV/JSON files in `data/` and `models/`.

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | All pipeline code (see the run order below). `pathsnap_core.py` holds the shared candidate and feature builder used by training and the API. `hmm.py` holds the HMM/Viterbi decoder. |
| `api/` | FastAPI backend (`api/main.py`). Request logs go to `api/logs/api.log` and the console. |
| `ui/` | Evaluator web UI (`ui/app.py`). |
| `models/` | Final tuned classifier (`histgb_tuned.joblib`), feature list, HMM decoder configuration (`hmm_config.json`). |
| `data/` | Experiment outputs: inventory, candidate summary, labels, model comparison, CV folds, Optuna trials, tuned vs untuned, HMM results. |
| `figures/` | All report plots (made by `src/plots.py`). |
| `maps/`, `eda/` | Folium HTML maps. |
| `docs/` | Report and team notes (`CLAUDE.md`). |
| `Problem-Statement/` | Organiser documents and the student dataset (train 80 trips, student_test 10 trips). |
| `server_data/` | **Not in git.** Evaluator-test trips, which exist only on the API laptop (see Deployment). |

## Installation

Python 3.12 to 3.14.

```bash
python -m venv .venv
```

```bash
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the experiments (run from the repository root)

| Step | Command | Main outputs |
|---|---|---|
| 1. Raw-data inventory and checks | `python src/data_processing.py` | `data/dataset_inventory.csv`, `data/network_connectivity.csv` |
| 2. EDA, label derivation check, Folium EDA map | `python src/folium_review.py` | `data/trip_stats.csv`, `eda/*.html` |
| 3. Candidates and features | `python src/feature.py` (about 1 min) | `data/features_*.pkl` (git-ignored), `data/candidate_build_summary.csv`, `data/point_labels.csv` |
| 4. Seven-model comparison (GroupKFold) | `python src/models.py` | `data/model_comparison.csv`, `data/cv_fold_scores.csv` |
| 5. Bayesian optimisation (Optuna TPE) | `python src/tune_histgb.py` (about 30 min) | `data/optuna_histgb_trials.csv`, `data/optuna_histgb_best.json`, `data/tuned_vs_untuned.csv` |
| 6. Final model, HMM decoder tuning and evaluation | `python src/hmm_experiment.py` (about 15 min) | `models/histgb_tuned.joblib`, `models/hmm_config.json`, `data/hmm_*.csv`, `data/student_test_predictions.csv` |
| 7. Folium maps of the student-test trips | `python src/make_maps.py` | `maps/student_test_*.html` |
| 8. Report figures | `python src/plots.py` | `figures/*.png` |

All randomness uses seed 42. The evaluator-test trips are never read by steps 1 to 8.

## Deployment (two laptops)

Both laptops need this repository and `pip install -r requirements.txt`, and both must be on the same network.

### Laptop 1: model and API

1. Put the 10 evaluator trips in `server_data/evaluator_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}`. For example, unzip `evaluator_test.zip` into `server_data/`. You can also point `PATHSNAP_EVAL_DIR` to another folder.
2. Start the API (the logs appear in this console, which is what you show the evaluator):

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

3. Find the laptop's IP address (`ipconfig` on Windows, `ip addr` on Linux, `ifconfig` on macOS). Allow port 8000 through the firewall.

Endpoints:
- `GET /health`: model status.
- `GET /trips`: the evaluation trip ids.
- `POST /predict` with body `{"trip_id": "00000015"}`: returns the predicted arc per GPS point, the predicted route geometry, nearby road-network geometry, and the server-computed metrics. The metrics are Top-1 accuracy, route continuity and route precision/recall/F1, for classifier + HMM and for classifier only.
- Interactive docs are at `http://<laptop-1-ip>:8000/docs`.

Ground truth stays on the server. The `.route` file is read only in `score()` in `api/main.py`, after the prediction is finished, and the response carries metrics only, never reference arcs or geometry.

### Laptop 2: UI

```bash
# Windows PowerShell:  $env:PATHSNAP_API_URL="http://<laptop-1-ip>:8000"
export PATHSNAP_API_URL=http://<laptop-1-ip>:8000
uvicorn ui.app:app --host 0.0.0.0 --port 8501
```

Open `http://localhost:8501`. Click any of the 10 trips. The UI sends only the trip id to the API and then shows:
- the metrics table, with classifier + HMM next to classifier only;
- a Folium map with a layer control and a legend. Its layers are the road network, raw GPS, the predicted route and the classifier-only route.

To test on a single machine, run both commands on the same laptop. The default API URL is `http://127.0.0.1:8000`.

## Rules we follow

- The evaluator-test data is never used for training, features, thresholds, model choice or tuning. It is not committed to the repository.
- Splits are by trip only (GroupKFold with trip as the group). All learned preprocessing is fitted on train trips only.
- No predictions are hardcoded. The API runs the full pipeline on every request and logs every request.
- AI assistance (Claude) was used for parts of the code and documentation. It is disclosed in the report.
