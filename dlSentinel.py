import os
import shutil
from collections import defaultdict
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.errors import GEOSException

from pystac_client import Client
import planetary_computer as pc

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.features import rasterize


# =========================
# CONFIG
# =========================
GPKG_PATH = "Parcelles-Herault.gpkg"
ID_COL = "ID_PARCEL"

OUTROOT = "data/s2_herault_2024_full"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

START, END = "2024-01-01", "2024-12-31"

# Sur l'année entière, monter le seuil nuageux est utile (sinon hiver = rien).
MAX_CLOUD = 80

# Fenêtrage temporel (aligné revisite)
WINDOW_DAYS = 5          # 5 jours recommandé
TOP_K_PER_WINDOW = 1     # 1 scène par fenêtre (tu peux mettre 2 si tu veux plus dense)

# Bandes nécessaires + SCL (masque nuages)
# Indices de base: NDVI (B04,B08), NDMI (B08,B11), NDWI (B03,B08), EVI (B02,B04,B08)
BANDS = ["B02", "B03", "B04", "B08", "B11", "SCL"]

# Indices exportés dans le CSV long.
# On garde les 4 historiques + des indices supplémentaires
# calculables avec B02/B03/B04/B08/B11.
INDEX_NAMES = [
    "NDVI",
    "NDMI",
    "NDWI",
    "EVI",
    "GNDVI",
    "SAVI",
    "OSAVI",
    "MSAVI",
    "MNDWI",
    "ARVI",
    "BSI",
]

IDX_NODATA = -9999.0
PAD_DEG = 0.0
MIN_PX_COUNT = 10   # filtrage qualité (0 = tout garder)
# Supprime les rasters temporaires de chaque scène pour limiter l'espace disque.
KEEP_SCENE_RASTERS = False

# SCL classes à masquer (nuages/ombres/neige/nodata)
# SCL classes (ESA):
# 0 No data, 1 Saturated/defective, 2 Dark features, 3 Cloud shadows,
# 4 Vegetation, 5 Bare soils, 6 Water, 7 Unclassified,
# 8 Cloud med prob, 9 Cloud high prob, 10 Thin cirrus, 11 Snow/ice
SCL_MASK_VALUES = {0, 1, 3, 8, 9, 10, 11}

os.makedirs(OUTROOT, exist_ok=True)


# =========================
# Utils dates
# =========================
def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def window_start(d: date, start_date: date, window_days: int) -> date:
    delta = (d - start_date).days
    k = (delta // window_days) * window_days
    return start_date + timedelta(days=k)


# =========================
# 1) Charger parcelles + fix geom
# =========================
gdf = gpd.read_file(GPKG_PATH)
if gdf.crs is None:
    raise ValueError("CRS manquant dans le GeoPackage.")
if ID_COL not in gdf.columns:
    raise ValueError(f"Colonne ID '{ID_COL}' introuvable. Colonnes: {list(gdf.columns)}")

gdf_4326 = gdf.to_crs(4326)
gdf_4326 = gdf_4326[~gdf_4326.geometry.is_empty & gdf_4326.geometry.notna()].copy()

invalid = ~gdf_4326.geometry.is_valid
print("Géométries invalides:", int(invalid.sum()), "sur", len(gdf_4326))
if invalid.any():
    gdf_4326.loc[invalid, "geometry"] = gdf_4326.loc[invalid, "geometry"].buffer(0)

invalid2 = ~gdf_4326.geometry.is_valid
print("Encore invalides après fix:", int(invalid2.sum()))
if invalid2.any():
    gdf_4326 = gdf_4326.loc[~invalid2].copy()

minx, miny, maxx, maxy = gdf_4326.total_bounds
dept_bbox_4326 = (minx, miny, maxx, maxy)

print("Parcelles valides:", len(gdf_4326))
print("BBox département (4326):", dept_bbox_4326)


# =========================
# 2) STAC search (2024 entier)
# =========================
catalog = Client.open(STAC_URL)
search = catalog.search(
    collections=[COLLECTION],
    bbox=list(dept_bbox_4326),
    datetime=f"{START}/{END}",
    query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
)
items = list(search.items())
print(f"\nNb items S2 L2A trouvés ({START}→{END}, cloud<{MAX_CLOUD}%): {len(items)}")
if not items:
    raise RuntimeError("Aucun item trouvé. Augmente MAX_CLOUD ou élargis la période.")


# =========================
# 3) Grouper par tuile
# =========================
by_tile = defaultdict(list)
for it in items:
    tile = it.properties.get("s2:mgrs_tile")
    if tile:
        by_tile[tile].append(it)

tiles = sorted(by_tile.keys())
print("Tuiles MGRS détectées:", tiles)


# =========================
# 4) Fonctions footprint + sélection coverage-first
# =========================
def parcels_in_item_footprint(gdf_4326, item, pad_deg=0.0):
    foot = shape(item.geometry)

    # préfiltre bbox
    minx, miny, maxx, maxy = foot.bounds
    cand = gdf_4326.cx[minx:maxx, miny:maxy]
    if cand.empty:
        return None, None

    try:
        sub = cand[cand.intersects(foot)].copy()
    except GEOSException:
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


def coverage_count(gdf_4326, item):
    _, sub = parcels_in_item_footprint(gdf_4326, item, pad_deg=0.0)
    return 0 if sub is None else int(len(sub))


def pick_best_items_coverage_first(gdf_4326, candidates, top_k=1):
    """
    Retourne top_k items triés par:
      1) couverture (nb parcelles intersectées) décroissant
      2) cloud_cover croissant
    """
    scored = []
    for it in candidates:
        cov = coverage_count(gdf_4326, it)
        if cov <= 0:
            continue
        cc = float(it.properties.get("eo:cloud_cover", 100.0))
        scored.append((cov, cc, it))
    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [x[2] for x in scored[:top_k]]


# =========================
# 5) Sélection: par tuile + fenêtre 5 jours (coverage-first, cloud-second)
# =========================
start_date = parse_date(START)
end_date = parse_date(END)

selected = defaultdict(list)  # tile -> list(items sélectionnés)

for tile, its in by_tile.items():
    # regroupe par fenêtre
    bins = defaultdict(list)
    for it in its:
        d = it.datetime.date()
        ws = window_start(d, start_date, WINDOW_DAYS)
        bins[ws].append(it)

    print(f"\nTile {tile}: {len(its)} items -> {len(bins)} fenêtres de {WINDOW_DAYS} jours")
    for ws, cand in sorted(bins.items(), key=lambda kv: kv[0]):
        picked = pick_best_items_coverage_first(gdf_4326, cand, top_k=TOP_K_PER_WINDOW)
        for it in picked:
            selected[tile].append(it)

    print(f"  -> sélectionnés: {len(selected[tile])} items")

# résumé
total_sel = sum(len(v) for v in selected.values())
print(f"\nTotal items sélectionnés (toutes tuiles): {total_sel}")


# =========================
# 6) Download + crop + align + indices + zonal fast (multi-indices)
# =========================
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


def align_to_ref(src_path, ref_path, out_path, resampling=Resampling.nearest):
    """
    Reprojette/resample src sur la grille de ref (même crs/transform/shape).
    Utile pour B11(20m) et SCL(20m) -> grille B08(10m).
    """
    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_h, dst_w = ref.height, ref.width
        dst_profile = ref.profile.copy()
        dst_profile.update(count=1, dtype="float32")

    with rasterio.open(src_path) as src:
        src_arr = src.read(1)
        dst_arr = np.zeros((dst_h, dst_w), dtype=np.float32)

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **dst_profile) as dst:
        dst.write(dst_arr.astype(np.float32), 1)

    return out_path


def write_index(out_path, ref_path, arr_float32, nodata_val=IDX_NODATA):
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        profile.update(dtype="float32", count=1, nodata=nodata_val, compress="deflate")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr_float32.astype(np.float32), 1)
    return out_path


def compute_indices_with_scl(band_paths, out_dir, nodata_val=IDX_NODATA):
    """
    Calcule NDVI, NDMI, NDWI, EVI sur la grille B08 (10m),
    en masquant nuages via SCL (reprojetté sur B08).
    """
    b02 = band_paths["B02"]
    b03 = band_paths["B03"]
    b04 = band_paths["B04"]
    b08 = band_paths["B08"]
    b11 = band_paths["B11"]
    scl = band_paths["SCL"]

    # align B11 & SCL -> grille B08 (10m)
    b11_10m = os.path.join(out_dir, "B11_10m.tif")
    if not os.path.exists(b11_10m):
        align_to_ref(b11, b08, b11_10m, resampling=Resampling.nearest)

    scl_10m = os.path.join(out_dir, "SCL_10m.tif")
    if not os.path.exists(scl_10m):
        # nearest obligatoire pour classes
        align_to_ref(scl, b08, scl_10m, resampling=Resampling.nearest)

    # lire bandes
    with rasterio.open(b08) as s8: nir = s8.read(1).astype(np.float32)
    with rasterio.open(b04) as s4: red = s4.read(1).astype(np.float32)
    with rasterio.open(b03) as s3: green = s3.read(1).astype(np.float32)
    with rasterio.open(b02) as s2: blue = s2.read(1).astype(np.float32)
    with rasterio.open(b11_10m) as s11: swir = s11.read(1).astype(np.float32)
    with rasterio.open(scl_10m) as sscl: scl_arr = sscl.read(1).astype(np.int16)

    # masques
    m0 = (nir == 0) & (red == 0)
    mcloud = np.isin(scl_arr, np.array(list(SCL_MASK_VALUES), dtype=np.int16))
    mmask = m0 | mcloud

    # indices
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndmi = (nir - swir) / (nir + swir + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)
    evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + 1e-6)
    gndvi = (nir - green) / (nir + green + 1e-6)
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + 1e-6)
    osavi = 1.16 * (nir - red) / (nir + red + 0.16 + 1e-6)
    msavi_root = np.maximum((2.0 * nir + 1.0) ** 2 - 8.0 * (nir - red), 0.0)
    msavi = 0.5 * ((2.0 * nir + 1.0) - np.sqrt(msavi_root))
    mndwi = (green - swir) / (green + swir + 1e-6)
    arvi = (nir - (2.0 * red - blue)) / (nir + (2.0 * red - blue) + 1e-6)
    bsi = ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue) + 1e-6)

    # nodata
    for a in (ndvi, ndmi, ndwi, evi, gndvi, savi, osavi, msavi, mndwi, arvi, bsi):
        a[mmask] = nodata_val

    out = {}
    all_idx = {
        "NDVI": ndvi,
        "NDMI": ndmi,
        "NDWI": ndwi,
        "EVI": evi,
        "GNDVI": gndvi,
        "SAVI": savi,
        "OSAVI": osavi,
        "MSAVI": msavi,
        "MNDWI": mndwi,
        "ARVI": arvi,
        "BSI": bsi,
    }
    for idx_name in INDEX_NAMES:
        if idx_name not in all_idx:
            continue
        out[idx_name] = write_index(
            os.path.join(out_dir, f"{idx_name}.tif"),
            b08,
            all_idx[idx_name],
            nodata_val,
        )
    return out


def fast_zonal_multi_mean_count(parcels_gdf, parcels_id_col, raster_paths_dict, nodata_val, progress_every=80):
    """
    FAST zonal stats pour plusieurs rasters (mean+count) en UNE seule passe.
    - rasterize une fois les parcelles
    - parcourt les blocks et accumulate sum/count pour chaque index
    Retourne df: ID, <idx>_mean, <idx>_count
    """
    idx_names = list(raster_paths_dict.keys())
    paths = [raster_paths_dict[k] for k in idx_names]

    # On prend le 1er raster comme référence grille/blocks
    with rasterio.open(paths[0]) as ref:
        transform = ref.transform
        out_shape = (ref.height, ref.width)
        block_windows = list(ref.block_windows(1))

    parcels = parcels_gdf[[parcels_id_col, "geometry"]].copy().reset_index(drop=True)
    ids = parcels[parcels_id_col].astype(str).values
    int_ids = np.arange(1, len(parcels) + 1, dtype=np.int32)

    print("    Rasterize parcels -> label raster...")
    shapes = [(geom, int(i)) for geom, i in zip(parcels.geometry.values, int_ids)]
    lab = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="int32",
    )

    # accumulateurs par index
    sums = {k: np.zeros(len(parcels) + 1, dtype=np.float64) for k in idx_names}
    counts = {k: np.zeros(len(parcels) + 1, dtype=np.int64) for k in idx_names}

    # ouvrir tous les rasters
    srcs = [rasterio.open(p) for p in paths]
    try:
        print("    Accumulate mean/count by blocks (multi-indices)...")
        for bi, (_, win) in enumerate(block_windows, start=1):
            lab_w = lab[win.row_off:win.row_off + win.height, win.col_off:win.col_off + win.width]
            if not np.any(lab_w > 0):
                if progress_every and bi % progress_every == 0:
                    print(f"      blocks processed: {bi}")
                continue

            for idx_name, src in zip(idx_names, srcs):
                data = src.read(1, window=win).astype(np.float32)
                m = (lab_w > 0) & (data != nodata_val)
                if np.any(m):
                    pid = lab_w[m].astype(np.int32)
                    val = data[m].astype(np.float32)
                    counts[idx_name] += np.bincount(pid, minlength=counts[idx_name].size)
                    sums[idx_name] += np.bincount(pid, weights=val, minlength=sums[idx_name].size)

            if progress_every and bi % progress_every == 0:
                print(f"      blocks processed: {bi}")

    finally:
        for s in srcs:
            s.close()

    # construire df
    out = pd.DataFrame({parcels_id_col: ids})
    for idx_name in idx_names:
        mean = np.full(len(parcels) + 1, np.nan, dtype=np.float64)
        ok = counts[idx_name] > 0
        mean[ok] = sums[idx_name][ok] / counts[idx_name][ok]
        out[f"{idx_name.lower()}_mean"] = mean[1:]
        out[f"{idx_name.lower()}_count"] = counts[idx_name][1:]

    return out


# =========================
# 7) Traitement principal
# =========================
rows = []

for tile, its in selected.items():
    print(f"\n===== PROCESS TILE {tile}: {len(its)} dates sélectionnées =====")
    its_sorted = sorted(its, key=lambda it: it.datetime)

    for it in its_sorted:
        d = it.datetime.date().isoformat()
        cloud = float(it.properties.get("eo:cloud_cover", 100.0))
        item_id = it.id

        # footprint -> parcelles + bbox crop
        crop_bbox, parcels_tile = parcels_in_item_footprint(gdf_4326, it, pad_deg=PAD_DEG)
        if crop_bbox is None:
            continue

        out_dir = os.path.join(OUTROOT, f"{d}_{tile}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n--- {tile} | {d} | cloud={cloud:.1f}% | parcels={len(parcels_tile)} ---")

        # download + crop bandes
        band_paths = {}
        ok = True
        for b in BANDS:
            if b not in it.assets:
                print(f"  ⚠️ Asset manquant {b} pour {item_id} -> skip")
                ok = False
                break
            out_path = os.path.join(out_dir, f"{b}.tif")
            if not os.path.exists(out_path):
                try:
                    p = download_crop_band(it.assets[b].href, out_path, crop_bbox)
                except Exception as e:
                    print(f"  ⚠️ Échec écriture raster {b} ({tile} {d}): {e} -> skip scène")
                    ok = False
                    break
                if p is None:
                    ok = False
                    break
            band_paths[b] = out_path

        if not ok:
            if not KEEP_SCENE_RASTERS:
                shutil.rmtree(out_dir, ignore_errors=True)
            continue

        # indices + mask nuages via SCL
        try:
            idx_paths = compute_indices_with_scl(band_paths, out_dir, nodata_val=IDX_NODATA)
        except Exception as e:
            print(f"  ⚠️ Échec calcul indices ({tile} {d}): {e} -> skip scène")
            if not KEEP_SCENE_RASTERS:
                shutil.rmtree(out_dir, ignore_errors=True)
            continue
        print("  Indices:", ", ".join(idx_paths.keys()))

        # reproj parcelles au CRS (B08)
        with rasterio.open(band_paths["B08"]) as src:
            crs_ref = src.crs
        parcels_tile_r = parcels_tile.to_crs(crs_ref)

        # zonal fast multi-indices
        df_stats = fast_zonal_multi_mean_count(
            parcels_tile_r,
            parcels_id_col=ID_COL,
            raster_paths_dict=idx_paths,
            nodata_val=IDX_NODATA,
            progress_every=100
        )

        # filtrage qualité (au moins MIN_PX_COUNT sur NDVI, par exemple)
        if MIN_PX_COUNT > 0:
            ref_col = "ndvi_count" if "ndvi_count" in df_stats.columns else f"{INDEX_NAMES[0].lower()}_count"
            if ref_col in df_stats.columns:
                df_stats = df_stats[df_stats[ref_col].fillna(0) >= MIN_PX_COUNT].copy()

        # export format long (comme tu fais déjà)
        # une ligne par (parcelle, date, indice)
        for _, r in df_stats.iterrows():
            pid = r[ID_COL]
            for idx_name in idx_paths.keys():
                rows.append({
                    "date": d,
                    "tile": tile,
                    ID_COL: pid,
                    "index": idx_name,
                    "value_mean": r[f"{idx_name.lower()}_mean"],
                    "px_count": int(r[f"{idx_name.lower()}_count"]),
                    "cloud_scene": cloud,
                })

        if not KEEP_SCENE_RASTERS:
            shutil.rmtree(out_dir, ignore_errors=True)

# =========================
# 8) Export CSV final
# =========================
df = pd.DataFrame(rows)
out_csv = os.path.join(OUTROOT, f"indices_parcelles_{START}_{END}_win{WINDOW_DAYS}d.csv")
df.to_csv(out_csv, index=False)

print("\n✅ CSV écrit:", out_csv)
print("Nb lignes:", len(df))
if len(df):
    print(df.head())
