"""Folium map builder shared by the offline map script (make_maps.py) and the evaluator UI (ui/app.py).

Layers (each toggleable in the layer control):
  * Road network (relevant arcs near the trip)          grey thin lines
  * Raw GPS observations                                 blue dots
  * Reference route (only for development trips)         green thick line
  * Predicted route (classifier + HMM)                   red line
  * Classifier-only prediction (optional)                orange dashed line
  * Start / end markers
Coordinates in all inputs are [lat, lon] pairs (Folium order).
"""
import folium
from branca.element import Element

def add_basemaps(m):
    """Three basemap options (radio buttons in the layer control). OpenStreetMap's own tile servers often return
    403 for pages opened from file:// (no Referer header), so Esri streets is the default and the others are fallbacks."""
    folium.TileLayer("Esri.WorldStreetMap", name="Esri streets").add_to(m)
    folium.TileLayer("CartoDB positron", name="CartoDB light").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    return m


COLORS = dict(network="#9e9e9e", gps="#1f6fd1", reference="#1a9850", predicted="#d7301f", classifier="#fd8d3c")


def _legend(items):
    rows = "".join(f'<div><span style="display:inline-block;width:18px;height:4px;background:{c};margin-right:6px;'
                   f'vertical-align:middle"></span>{name}</div>' for name, c in items)
    return Element(f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
                   f'border:1px solid #888;border-radius:6px;font:13px sans-serif;line-height:1.6">'
                   f'<b>Legend</b>{rows}</div>')


def build_map(gps, network=None, predicted=None, reference=None, classifier=None, title=None, max_gps=6000):
    """gps: [[lat, lon], ...]; network: list of segments [[lat,lon],[lat,lon]]; predicted / reference / classifier:
    list of polylines (each a list of [lat, lon]) or a single polyline."""
    def as_lines(x):
        if not x: return []
        return x if isinstance(x[0][0], (list, tuple)) else [x]

    lats = [p[0] for p in gps]; lons = [p[1] for p in gps]
    m = add_basemaps(folium.Map(tiles=None, control_scale=True))
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(20, 20))
    legend = []
    if network:
        fg = folium.FeatureGroup(name="Road network (nearby arcs)", show=True)
        folium.PolyLine(network, color=COLORS["network"], weight=1.5, opacity=0.6).add_to(fg)  # MultiPolyLine
        fg.add_to(m); legend.append(("Road network", COLORS["network"]))
    if reference:
        fg = folium.FeatureGroup(name="Reference route", show=True)
        for line in as_lines(reference):
            folium.PolyLine(line, color=COLORS["reference"], weight=8, opacity=0.55).add_to(fg)
        fg.add_to(m); legend.append(("Reference route", COLORS["reference"]))
    if classifier:
        fg = folium.FeatureGroup(name="Classifier-only prediction", show=False)
        for line in as_lines(classifier):
            folium.PolyLine(line, color=COLORS["classifier"], weight=3, opacity=0.9, dash_array="6,6").add_to(fg)
        fg.add_to(m); legend.append(("Classifier-only", COLORS["classifier"]))
    if predicted:
        fg = folium.FeatureGroup(name="Predicted route (classifier + HMM)", show=True)
        for line in as_lines(predicted):
            folium.PolyLine(line, color=COLORS["predicted"], weight=4, opacity=0.9).add_to(fg)
        fg.add_to(m); legend.append(("Predicted route (HMM)", COLORS["predicted"]))
    fg = folium.FeatureGroup(name="Raw GPS observations", show=True)
    step = max(1, len(gps) // max_gps)
    for p in gps[::step]:
        folium.CircleMarker(p, radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8, weight=0).add_to(fg)
    fg.add_to(m); legend.append(("Raw GPS" + (f" (every {step}th point)" if step > 1 else ""), COLORS["gps"]))
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    if title:
        m.get_root().html.add_child(Element(f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;'
                                            f'padding:4px 10px;border-radius:6px;font:bold 14px sans-serif">{title}</div>'))
    m.get_root().html.add_child(_legend(legend))
    return m
