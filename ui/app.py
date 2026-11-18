"""PathSnap evaluator UI (Laptop 2).

Shows the ten evaluation trips (list fetched from the API), sends the selected trip id to the API, and displays the
returned predicted route on a Folium map together with the server-computed metrics. The UI never reads trip files
and never receives reference routes: everything comes from the API response.

Run:  PATHSNAP_API_URL=http://<laptop-1-ip>:8000  uvicorn ui.app:app --host 0.0.0.0 --port 8501
Then open http://localhost:8501 in a browser on Laptop 2.
"""
import os, sys, html, time
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from route_map import build_map

API_URL = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")
app = FastAPI(title="PathSnap UI")

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PathSnap Evaluator</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f5f6f8;color:#222}}
header{{background:#1f3b63;color:#fff;padding:12px 20px}} header h1{{margin:0;font-size:20px}} header p{{margin:2px 0 0;font-size:13px;opacity:.85}}
main{{padding:16px 20px}} .trips a{{display:inline-block;margin:4px;padding:8px 12px;border-radius:6px;background:#fff;
border:1px solid #c8cdd5;text-decoration:none;color:#1f3b63;font-weight:600}} .trips a.sel{{background:#1f3b63;color:#fff}}
table{{border-collapse:collapse;background:#fff;margin:10px 0}} td,th{{border:1px solid #d5d9e0;padding:5px 10px;font-size:14px;text-align:right}}
th{{background:#eef1f5}} td:first-child,th:first-child{{text-align:left}} .err{{color:#b00020;font-weight:600}}
.map{{height:640px;border:1px solid #c8cdd5;background:#fff}} .map iframe{{width:100%;height:100%;border:0}} small{{color:#555}}
</style></head><body><header><h1>PathSnap: map-matching evaluator</h1>
<p>API: {api} &nbsp;|&nbsp; classifier: tuned HistGradientBoosting &nbsp;|&nbsp; decoder: HMM + Viterbi</p></header>
<main><h3>Evaluation trips</h3><div class="trips">{trips}</div>{body}</main></body></html>"""


def get_trips():
    r = requests.get(f"{API_URL}/trips", timeout=10); r.raise_for_status()
    return r.json()["trips"]


def trips_html(trips, sel=None):
    return "".join(f'<a class="{"sel" if t == sel else ""}" href="/trip/{t}">{t}</a>' for t in trips)


@app.get("/", response_class=HTMLResponse)
def index():
    try:
        trips = get_trips()
        body = "<p>Select a trip. The UI sends only the trip id to the API; the API runs the full pipeline and scores it.</p>"
    except Exception as e:
        trips, body = [], f'<p class="err">Cannot reach the API at {html.escape(API_URL)}: {html.escape(str(e))}</p>'
    return PAGE.format(api=html.escape(API_URL), trips=trips_html(trips), body=body)


@app.get("/trip/{trip_id}", response_class=HTMLResponse)
def show_trip(trip_id: str):
    try:
        trips = get_trips()
        t0 = time.time()
        r = requests.post(f"{API_URL}/predict", json={"trip_id": trip_id}, timeout=600)
        rtt = time.time() - t0
        if r.status_code != 200:
            raise RuntimeError(f"API returned {r.status_code}: {r.text[:300]}")
        res = r.json()
    except Exception as e:
        return PAGE.format(api=html.escape(API_URL), trips="", body=f'<p class="err">{html.escape(str(e))}</p>')

    m, c = res["metrics"]["hmm"], res["metrics"]["classifier_only"]
    rows = [("Observation-level Top-1 accuracy", "top1_accuracy"), ("Correct points", "correct_points"),
            ("Route continuity", "route_continuity"), ("Route precision (length-weighted)", "route_precision"),
            ("Route recall (length-weighted)", "route_recall"), ("Route F1", "route_f1")]
    table = "<table><tr><th>Metric (computed on the API server)</th><th>Classifier + HMM (final)</th><th>Classifier only</th></tr>"
    table += "".join(f"<tr><td>{n}</td><td><b>{m[k]}</b></td><td>{c[k]}</td></tr>" for n, k in rows)
    table += (f"<tr><td>GPS points / unmatched</td><td colspan=2>{res['n_points']} / {res['unmatched_points']}</td></tr>"
              f"<tr><td>API processing time (s)</td><td colspan=2>{res['timings']['total_s']} (round trip {rtt:.2f})</td></tr></table>")
    gps = [[p["lat"], p["lon"]] for p in res["points"]]
    fmap = build_map(gps, network=res["network"], predicted=res["predicted_route"], classifier=res["classifier_only_route"],
                     title=f"Trip {trip_id}")
    body = (f"<h3>Trip {html.escape(trip_id)}</h3>{table}<div class='map'>{fmap._repr_html_()}</div>"
            f"<small>Layers: road network near the trip, raw GPS, predicted route (HMM), classifier-only route (toggle on). "
            f"The reference route is kept on the API server and is only used there to compute the metrics above.</small>")
    return PAGE.format(api=html.escape(API_URL), trips=trips_html(trips, trip_id), body=body)
