import os, glob, collections, zipfile
import numpy as np
import pandas as pd

# ---------- 0. LOCATE DATA ----------
ROOT = os.path.join("Problem-Statement", "Student folder", "Student folder")

if not os.path.exists(ROOT):
    # fallback: maybe the zip is next to the script
    for cand in ["Student_folder.zip", "Student folder.zip",
                 os.path.join("Problem-Statement", "Student_folder.zip"),
                 os.path.join("Problem-Statement", "Student folder.zip")]:
        if os.path.exists(cand):
            with zipfile.ZipFile(cand) as z:
                z.extractall(".")
            break

print("ROOT exists:", os.path.exists(ROOT), "| cwd:", os.getcwd())
if not os.path.exists(ROOT):
    raise FileNotFoundError(f"Could not find data root: {ROOT}")

# Show the two expected splits
for split in ["train", "student_test"]:
    p = os.path.join(ROOT, split)
    print(f"  {split}: exists={os.path.exists(p)}")

# ---------- 1. FOLDER STRUCTURE ----------
print("\n=== FOLDER TREE (depth<=3, first 6 entries per folder) ===")
for root, dirs, files in os.walk(ROOT):
    depth = root[len(ROOT):].count(os.sep)
    if depth <= 3:
        print("  " * depth + os.path.basename(root) + "/", f"[{len(files)} files, {len(dirs)} subdirs]")
        for f in sorted(files)[:6]:
            print("  " * (depth + 1) + f, f"({os.path.getsize(os.path.join(root, f))} B)")

all_files = [p for p in glob.glob(f"{ROOT}/**/*", recursive=True) if os.path.isfile(p)]
print("\nExtension counts:", dict(collections.Counter(os.path.splitext(p)[1] for p in all_files)))
print("Total files found:", len(all_files))

# ---------- 2. GROUP FILES INTO TRIPS ----------
trips = collections.defaultdict(dict)
for p in all_files:
    base, ext = os.path.splitext(p)
    trips[base][ext] = p

print("Distinct trip keys (folder+basename):", len(trips))
print("Files-per-trip pattern:", dict(collections.Counter(tuple(sorted(v)) for v in trips.values())))

# Also flag train vs test membership (handy for later ML split work)
def split_of(base):
    rel = os.path.relpath(base, ROOT)
    return rel.split(os.sep)[0]
print("Trips per split:", dict(collections.Counter(split_of(b) for b in trips)))

# ---------- 3. RAW PEEK (first lines of one trip, every file type) ----------
sample = next(iter(sorted(trips)))
print("\n=== RAW PEEK:", sample, "===")
for ext, p in sorted(trips[sample].items()):
    with open(p, "r", errors="replace") as fh:
        head = [next(fh, "").rstrip() for _ in range(4)]
    print(ext, "->", head)

# ---------- 4. PER-TRIP INVENTORY ----------
def rd(p, cols, dt=float):
    return pd.read_csv(p, sep=r"\s+", header=None, names=cols, dtype=dt)

rows = []
for base, f in sorted(trips.items()):
    rel = os.path.relpath(base, ROOT)
    r = {
        "trip": rel,
        "split": rel.split(os.sep)[0],      # "train" or "student_test"
        "exts": ",".join(sorted(e.strip('.') for e in f)),
    }
    try:
        if ".track" in f:
            t = rd(f[".track"], ["lon", "lat", "ts"])
            dt_ = np.diff(t.ts.values)
            r.update(n_gps=len(t), nan=int(t.isna().sum().sum()),
                     dup_ts=int((dt_ == 0).sum()), neg_ts=int((dt_ < 0).sum()),
                     pct_1s=round(float((dt_ == 1).mean()) * 100, 1) if len(dt_) else np.nan,
                     max_gap_s=float(dt_.max()) if len(dt_) else np.nan,
                     lon_min=round(t.lon.min(), 5), lon_max=round(t.lon.max(), 5),
                     lat_min=round(t.lat.min(), 5), lat_max=round(t.lat.max(), 5))
        if ".nodes" in f:
            n = rd(f[".nodes"], ["lon", "lat"])
            r.update(n_nodes=len(n),
                     node_lon_min=round(n.lon.min(), 5), node_lon_max=round(n.lon.max(), 5),
                     node_lat_min=round(n.lat.min(), 5), node_lat_max=round(n.lat.max(), 5))
        if ".arcs" in f:
            a = rd(f[".arcs"], ["u", "v"], int)
            r.update(n_arcs=len(a), arc_max_node=int(max(a.u.max(), a.v.max())),
                     self_loops=int((a.u == a.v).sum()))
        if ".route" in f:
            rt = rd(f[".route"], ["arc"], int)
            r.update(n_route=len(rt), route_unique=rt.arc.nunique(), route_max_arc=int(rt.arc.max()))
    except Exception as e:
        r["error"] = str(e)[:80]
    rows.append(r)

inv = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
print("\n=== INVENTORY (first 12 rows) ===")
print(inv.head(12))
print("\n=== NUMERIC SUMMARY ===")
print(inv.describe().T[["count", "min", "mean", "max"]])
print("\nTrips missing a file type:", inv[inv.exts != inv.exts.mode()[0]][["trip", "exts"]].to_string())
inv.to_csv("dataset_inventory.csv", index=False)
print("\nWrote dataset_inventory.csv")

# ---------- 5. DO TRIPS SHARE ONE ROAD NETWORK OR HAVE THEIR OWN? ----------
if "n_nodes" in inv and "n_arcs" in inv:
    print("\nDistinct (n_nodes, n_arcs) combos across trips:",
          inv[["n_nodes", "n_arcs"]].drop_duplicates().shape[0], "of", len(inv), "trips")
    print("\nSplit-wise trip counts and file-type coverage:")
    print(inv.groupby("split")["exts"].value_counts())

# ---------- 6. RELATIONSHIP CHECKS ON ONE TRIP ----------
f = trips[sample]
if all(k in f for k in [".track", ".nodes", ".arcs"]):
    t = rd(f[".track"], ["lon", "lat", "ts"])
    n = rd(f[".nodes"], ["lon", "lat"])
    a = rd(f[".arcs"], ["u", "v"], int)
    print("\n=== RELATIONSHIP CHECKS:", sample, "===")
    print("GPS points:", len(t), "| nodes:", len(n), "| arcs:", len(a))
    print("Arc node ids valid (< n_nodes)?", bool((a[["u", "v"]].max().max()) < len(n)))
    print("GPS bbox inside node bbox? lon", t.lon.min() >= n.lon.min() and t.lon.max() <= n.lon.max(),
          "lat", t.lat.min() >= n.lat.min() and t.lat.max() <= n.lat.max())
    print("Coordinate order sanity (lon/lat ranges):", t.lon.min(), t.lon.max(), t.lat.min(), t.lat.max())

    lat0 = np.deg2rad(t.lat.mean())
    kx, ky = 111320 * np.cos(lat0), 110540
    P = np.c_[t.lon * kx, t.lat * ky]; N = np.c_[n.lon * kx, n.lat * ky]
    d = np.sqrt(((P[:, None, :] - N[None, :, :]) ** 2).sum(-1)).min(1) if len(P) * len(N) < 3e7 else None
    if d is not None:
        print("Dist GPS->nearest node (m): min/median/95%/max =",
              np.round(np.percentile(d, [0, 50, 95, 100]), 1))

    parent = list(range(len(n)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for u, v in a.values:
        ru, rv = find(u), find(v)
        if ru != rv: parent[ru] = rv
    comps = collections.Counter(find(i) for i in range(len(n)))
    deg = np.bincount(np.r_[a.u.values, a.v.values], minlength=len(n))
    print("Connected components:", len(comps), "| largest:", max(comps.values()),
          "| isolated nodes (deg 0):", int((deg == 0).sum()))
    print("Duplicate arcs:", int(a.duplicated().sum()), "| reverse-pair arcs present:",
          int(a.merge(a.rename(columns={"u": "v", "v": "u"}), on=["u", "v"]).shape[0]))

    if ".route" in f:
        rt = rd(f[".route"], ["arc"], int)
        print("Route arcs:", len(rt), "| all valid arc ids?", bool(rt.arc.max() < len(a)))
        ru_, rv_ = a.u.values[rt.arc.values], a.v.values[rt.arc.values]
        print("Route chain continuity (v[i]==u[i+1]):",
              round(float((rv_[:-1] == ru_[1:]).mean()), 3))
        print("Route arcs per GPS point ratio:", round(len(rt) / len(t), 3))