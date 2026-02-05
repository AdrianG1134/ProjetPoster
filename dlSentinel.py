import os
from collections import defaultdict
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.errors import GEOSException
from pystac_client import Client
import planetary_computer as pc
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterstats import zonal_stats

# =========================
# CONFIG
# =========================
GPKG_PATH = "Parcelles-Herault.gpkg"
ID_COL = "ID_PARCEL"

OUTROOT = "data/s2_herault_tiles_march_v3"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

START, END = "2023-03-01", "2023-03-31"
MAX_CLOUD = 30

BANDS_NDVI = ["B04", "B08"]
NDVI_NODATA = -9999.0
PAD_DEG = 0.0

os.makedirs(OUTROOT, exist_ok=True)

# =========================
# 1) Charger parcelles
# =========================
gdf = gpd.read_file(GPKG_PATH)
if gdf.crs is None:
    raise ValueError("CRS manquant dans le GeoPackage.")
if ID_COL not in gdf.columns:
    raise ValueError(f"Colonne ID '{ID_COL}' introuvable. Colonnes: {list(gdf.columns)}")

gdf_4326 = gdf.to_crs(4326)

# ---- Réparer géométries invalides (uniquement celles qui le sont) ----
# (1) enlève les géométries vides
gdf_4326 = gdf_4326[~gdf_4326.geometry.is_empty & gdf_4326.geometry.notna()].copy()

invalid = ~gdf_4326.geometry.is_valid
print("Géométries invalides:", int(invalid.sum()), "sur", len(gdf_4326))

if invalid.any():
    # Buffer(0) uniquement sur invalides
    gdf_4326.loc[invalid, "geometry"] = gdf_4326.loc[invalid, "geometry"].buffer(0)

# Re-check + drop si encore invalides
invalid2 = ~gdf_4326.geometry.is_valid
print("Encore invalides après fix:", int(invalid2.sum()))
if invalid2.any():
    gdf_4326 = gdf_4326.loc[~invalid2].copy()
    print("Après suppression invalides restantes:", len(gdf_4326))

# bbox département
minx, miny, maxx, maxy = gdf_4326.total_bounds
dept_bbox_4326 = (minx, miny, maxx, maxy)

print("Parcelles valides:", len(gdf_4326))
print("BBox département (4326):", dept_bbox_4326)

# =========================
# 2) Recherche STAC
# =========================
catalog = Client.open(STAC_URL)
search = catalog.search(
    collections=[COLLECTION],
    bbox=list(dept_bbox_4326),
    datetime=f"{START}/{END}",
    query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
)
items = list(search.get_items())
print(f"\nNb items S2 L2A trouvés ({START}→{END}, cloud<{MAX_CLOUD}%): {len(items)}")
if not items:
    raise RuntimeError("Aucun item trouvé. Augmente MAX_CLOUD ou élargis la période.")

# =========================
# 3) Grouper par tuile MGRS
# =========================
by_tile = defaultdict(list)
for it in items:
    tile = it.properties.get("s2:mgrs_tile")
    if tile:
        by_tile[tile].append(it)

tiles = sorted(by_tile.keys())
print("Tuiles MGRS détectées:", tiles)

# =========================
# 4) Best item par tuile (nuages min)
# =========================
best_per_tile = {}
for tile, its in by_tile.items():
    its_sorted = sorted(its, key=lambda it: it.properties.get("eo:cloud_cover", 100.0))
    best = its_sorted[0]
    best_per_tile[tile] = best
    print(f"Tile {tile}: best date={best.datetime.date()} cloud={best.properties.get('eo:cloud_cover', 100.0):.1f}% id={best.id}")

# =========================
# 5) Fonctions
# =========================
def parcels_in_item_footprint(gdf_4326, item, pad_deg=0.0):
    """
    Filtre les parcelles qui intersectent le footprint réel (item.geometry).
    On utilise sindex pour accélérer + on gère les GEOSException.
    """
    foot = shape(item.geometry)

    # accélération: pré-filtre via bounding box du footprint
    minx, miny, maxx, maxy = foot.bounds
    cand = gdf_4326.cx[minx:maxx, miny:maxy]

    if cand.empty:
        return None, None

    # intersects robuste (au cas où)
    try:
        sub = cand[cand.intersects(foot)].copy()
    except GEOSException:
        # fallback: buffer(0) sur un petit sous-ensemble candidat, puis intersects
        cand2 = cand.copy()
        bad = ~cand2.geometry.is_valid
        if bad.any():
            cand2.loc[bad, "geometry"] = cand2.loc[bad, "geometry"].buffer(0)
            cand2 = cand2[cand2.geometry.is_valid]
        sub = cand2[cand2.intersects(foot)].copy()

    if sub.empty:
        return None, None

    minx2, miny2, maxx2, maxy2 = sub.total_bounds
    if pad_deg and pad_deg > 0:
        minx2 -= pad_deg; miny2 -= pad_deg; maxx2 += pad_deg; maxy2 += pad_deg

    return (minx2, miny2, maxx2, maxy2), sub

def download_crop_band(href, out_path, bbox4326):
    signed = pc.sign(href)
    with rasterio.open(signed) as src:
        bb = transform_bounds("EPSG:4326", src.crs, *bbox4326, densify_pts=21)
        inter = (
            max(bb[0], src.bounds.left),
            max(bb[1], src.bounds.bottom),
            min(bb[2], src.bounds.right),
            min(bb[3], src.bounds.top),
        )
        if inter[0] >= inter[2] or inter[1] >= inter[3]:
            return None

        win = from_bounds(*inter, transform=src.transform).round_offsets().round_lengths()
        data = src.read(1, window=win)
        transform = src.window_transform(win)

        profile = src.profile.copy()
        profile.update(
            height=data.shape[0],
            width=data.shape[1],
            transform=transform,
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
    return out_path

def pct_nonzero(path):
    with rasterio.open(path) as src:
        a = src.read(1)
    return 100.0 * float(np.mean(a != 0))

def compute_ndvi(b08_path, b04_path, out_path, nodata_val=-9999.0):
    with rasterio.open(b08_path) as s8, rasterio.open(b04_path) as s4:
        nir = s8.read(1).astype("float32")
        red = s4.read(1).astype("float32")
        ndvi = (nir - red) / (nir + red + 1e-6)

        # nodata là où nir & red sont nodata (=0)
        mask = (nir == 0) & (red == 0)
        ndvi[mask] = nodata_val

        profile = s8.profile.copy()
        profile.update(dtype="float32", count=1, nodata=nodata_val, compress="deflate")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(ndvi.astype("float32"), 1)
    return out_path

# =========================
# 6) Boucle par tuile + stats
# =========================
rows = []

for tile, it in best_per_tile.items():
    date_str = it.datetime.date().isoformat()
    cloud = float(it.properties.get("eo:cloud_cover", 100.0))

    print(f"\n=== TILE {tile} | date={date_str} cloud={cloud:.1f}% ===")

    crop_bbox, parcels_tile = parcels_in_item_footprint(gdf_4326, it, pad_deg=PAD_DEG)
    if crop_bbox is None:
        print("Aucune parcelle intersecte le footprint réel -> skip.")
        continue

    print("Nb parcelles footprint:", len(parcels_tile))
    print("Crop bbox:", crop_bbox)

    out_dir = os.path.join(OUTROOT, f"{date_str}_{tile}")
    os.makedirs(out_dir, exist_ok=True)

    out_paths = {}
    for b in BANDS_NDVI:
        out_path = os.path.join(out_dir, f"{b}.tif")
        if not os.path.exists(out_path):
            p = download_crop_band(it.assets[b].href, out_path, crop_bbox)
            if p is None:
                print(f"  - {b}: pas d'intersection -> skip tile")
                out_paths = {}
                break
        out_paths[b] = out_path
        print(f"  - {b}: nonzero={pct_nonzero(out_path):.2f}%")

    if not out_paths:
        continue

    ndvi_path = os.path.join(out_dir, "NDVI.tif")
    if not os.path.exists(ndvi_path):
        compute_ndvi(out_paths["B08"], out_paths["B04"], ndvi_path, nodata_val=NDVI_NODATA)
    print("  - NDVI:", ndvi_path)

    # zonal stats
    with rasterio.open(ndvi_path) as src:
        ndvi_crs = src.crs

    parcels_tile_r = parcels_tile.to_crs(ndvi_crs)

    zs = zonal_stats(
        parcels_tile_r.geometry,
        ndvi_path,
        stats=["mean", "median", "count"],
        nodata=NDVI_NODATA,
        all_touched=False,
    )

    for pid, s in zip(parcels_tile_r[ID_COL].values, zs):
        rows.append({
            "date": date_str,
            "tile": tile,
            ID_COL: pid,
            "ndvi_mean": s.get("mean"),
            "ndvi_median": s.get("median"),
            "px_count": s.get("count"),
            "cloud_scene": cloud,
        })

# =========================
# 7) Export CSV
# =========================
df = pd.DataFrame(rows)
out_csv = os.path.join(OUTROOT, f"ndvi_parcelles_{START}_{END}.csv")
df.to_csv(out_csv, index=False)

print("\n✅ CSV écrit:", out_csv)
print("Nb lignes:", len(df))
if len(df):
    print(df.head())
    print("px_count==0:", int((df["px_count"].fillna(0) == 0).sum()))
