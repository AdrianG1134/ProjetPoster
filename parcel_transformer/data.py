from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, Sampler

from config import DataConfig


LOGGER = logging.getLogger(__name__)


@dataclass
class PreparedDataset:
    features: np.ndarray
    day_of_year: np.ndarray
    observed_mask: np.ndarray
    cloud_scene: Optional[np.ndarray]
    px_count: Optional[np.ndarray]
    labels: np.ndarray
    parcel_ids: np.ndarray
    parcel_tiles: np.ndarray
    feature_names: list[str]
    label_names: list[str]
    time_grid: np.ndarray
    splits: dict[str, np.ndarray]
    group_labels: Optional[np.ndarray] = None
    group_label_names: list[str] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    @property
    def seq_len(self) -> int:
        return self.features.shape[1]

    @property
    def has_group_labels(self) -> bool:
        return self.group_labels is not None and len(self.group_label_names) > 0

    @property
    def num_group_classes(self) -> int:
        return len(self.group_label_names)


@dataclass
class TemporalAugmentationConfig:
    time_mask_ratio: float = 0.0
    jitter_std: float = 0.0


class ParcelTimeSeriesDataset(Dataset):
    def __init__(
        self,
        prepared: PreparedDataset,
        indices: np.ndarray,
        augmentation: Optional[TemporalAugmentationConfig] = None,
    ) -> None:
        self.features = prepared.features[indices]
        self.day_of_year = prepared.day_of_year[indices]
        self.observed_mask = prepared.observed_mask[indices]
        default_cloud = np.full(self.day_of_year.shape, 100.0, dtype=np.float32)
        default_px = np.zeros(self.day_of_year.shape, dtype=np.float32)
        self.cloud_scene = (
            prepared.cloud_scene[indices].astype(np.float32)
            if prepared.cloud_scene is not None
            else default_cloud
        )
        self.px_count = (
            prepared.px_count[indices].astype(np.float32)
            if prepared.px_count is not None
            else default_px
        )
        self.labels = prepared.labels[indices]
        self.group_labels = (
            prepared.group_labels[indices]
            if prepared.group_labels is not None
            else np.full((len(indices),), -100, dtype=np.int64)
        )
        self.parcel_ids = prepared.parcel_ids[indices]
        self.augmentation = augmentation

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        features = self.features[idx]
        observed_mask = self.observed_mask[idx]
        cloud_scene = self.cloud_scene[idx]
        px_count = self.px_count[idx]

        if self.augmentation is not None:
            features = features.copy()
            observed_mask = observed_mask.copy()
            cloud_scene = cloud_scene.copy()
            px_count = px_count.copy()

            observed_idx = np.flatnonzero(observed_mask)
            if self.augmentation.time_mask_ratio > 0 and observed_idx.size > 0:
                n_mask = int(round(observed_idx.size * self.augmentation.time_mask_ratio))
                n_mask = max(1, min(observed_idx.size, n_mask))
                masked_steps = np.random.choice(observed_idx, size=n_mask, replace=False)
                features[masked_steps, :] = 0.0
                observed_mask[masked_steps] = False

            if self.augmentation.jitter_std > 0:
                noise = np.random.normal(
                    loc=0.0,
                    scale=self.augmentation.jitter_std,
                    size=features.shape,
                ).astype(np.float32)
                features[observed_mask] = features[observed_mask] + noise[observed_mask]

        cloud_norm = np.clip(cloud_scene / 100.0, 0.0, 1.0).astype(np.float32)
        px_norm = np.clip(np.log1p(np.clip(px_count, a_min=0.0, a_max=None)) / np.log1p(256.0), 0.0, 1.0).astype(
            np.float32
        )
        cloud_norm = np.where(observed_mask, cloud_norm, 1.0).astype(np.float32)
        px_norm = np.where(observed_mask, px_norm, 0.0).astype(np.float32)
        reliability = (
            observed_mask.astype(np.float32)
            * (1.0 - cloud_norm)
            * np.sqrt(np.clip(px_norm, a_min=0.0, a_max=1.0))
        ).astype(np.float32)
        quality_features = np.stack([reliability, cloud_norm, px_norm], axis=-1).astype(np.float32)

        return {
            "features": torch.from_numpy(features),
            "day_of_year": torch.from_numpy(self.day_of_year[idx]).float(),
            "observed_mask": torch.from_numpy(observed_mask),
            "quality_features": torch.from_numpy(quality_features),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "group_label": torch.tensor(int(self.group_labels[idx]), dtype=torch.long),
            "parcel_id": str(self.parcel_ids[idx]),
        }


def _bin_dates_to_frequency(dates: pd.Series, frequency: str) -> pd.Series:
    """
    Bin timestamps to a regular temporal grid.

    For day-based frequencies (e.g. 5D), bins are anchored on the dataset
    minimum date to avoid calendar-epoch artifacts.
    """
    freq = str(frequency).strip()
    if not freq:
        return dates

    match = re.fullmatch(r"(?i)(\d+)D", freq)
    if match:
        step_days = int(match.group(1))
        if step_days <= 0:
            raise ValueError(f"Invalid day frequency: {frequency}")
        base_date = pd.to_datetime(dates.min()).normalize()
        normalized = pd.to_datetime(dates).dt.normalize()
        delta_days = (normalized - base_date).dt.days
        bin_start = base_date + pd.to_timedelta((delta_days // step_days) * step_days, unit="D")
        return pd.to_datetime(bin_start)

    # Fallback for non-day frequencies.
    try:
        return pd.to_datetime(dates).dt.floor(freq)
    except (ValueError, TypeError):
        return pd.to_datetime(dates).dt.to_period(freq).dt.start_time


def _required_columns(cfg: DataConfig) -> list[str]:
    return [
        cfg.parcel_id_col,
        cfg.date_col,
        cfg.index_col,
        cfg.value_col,
        cfg.px_count_col,
        cfg.cloud_col,
        cfg.tile_col,
        cfg.label_col,
    ]


def _mode_or_first(series: pd.Series) -> str:
    series = series.dropna().astype(str).str.strip()
    series = series[series != ""]
    if series.empty:
        return ""
    mode = series.mode(dropna=True)
    if not mode.empty:
        return str(sorted(mode.astype(str).tolist())[0])
    return str(series.iloc[0])


def _validate_input_dataframe(df: pd.DataFrame, cfg: DataConfig) -> None:
    missing = [col for col in _required_columns(cfg) if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}. "
            f"Expected columns include {', '.join(_required_columns(cfg))}."
        )


def load_long_dataframe(cfg: DataConfig) -> pd.DataFrame:
    csv_path = Path(cfg.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [col for col in _required_columns(cfg) if col not in header_cols]
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}. "
            f"Expected columns include {', '.join(_required_columns(cfg))}."
        )

    usecols = list(_required_columns(cfg))
    if cfg.label_group_col in header_cols and cfg.label_group_col not in usecols:
        usecols.append(cfg.label_group_col)

    str_dtype_cols = [cfg.parcel_id_col, cfg.index_col, cfg.label_col, cfg.tile_col]
    if cfg.label_group_col in usecols:
        str_dtype_cols.append(cfg.label_group_col)
    dtype_map = {col: "string" for col in str_dtype_cols}

    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=dtype_map,
        low_memory=False,
    )
    _validate_input_dataframe(df, cfg)

    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    if df[cfg.date_col].isna().all():
        raise ValueError(
            f"Column '{cfg.date_col}' could not be parsed as dates. "
            "Use an ISO format such as YYYY-MM-DD."
        )

    df = df.dropna(
        subset=[cfg.parcel_id_col, cfg.date_col, cfg.index_col, cfg.value_col, cfg.label_col]
    ).copy()

    df[cfg.parcel_id_col] = (
        df[cfg.parcel_id_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
    )
    df[cfg.index_col] = df[cfg.index_col].astype(str).str.strip().str.upper()
    df[cfg.label_col] = df[cfg.label_col].astype(str).str.strip()
    df[cfg.tile_col] = df[cfg.tile_col].astype(str).str.strip()
    if cfg.label_group_col in df.columns:
        df[cfg.label_group_col] = df[cfg.label_group_col].astype(str).str.strip()
    df[cfg.value_col] = pd.to_numeric(df[cfg.value_col], errors="coerce")
    df[cfg.px_count_col] = pd.to_numeric(df[cfg.px_count_col], errors="coerce")
    df[cfg.cloud_col] = pd.to_numeric(df[cfg.cloud_col], errors="coerce")
    df = df.dropna(subset=[cfg.value_col]).copy()
    df = df[
        (df[cfg.parcel_id_col] != "")
        & (df[cfg.index_col] != "")
        & (df[cfg.label_col] != "")
        & (df[cfg.tile_col] != "")
    ].copy()

    if cfg.index_filter:
        allowed = {str(x).strip().upper() for x in cfg.index_filter if str(x).strip()}
        available = set(df[cfg.index_col].dropna().astype(str).unique().tolist())
        missing = sorted(allowed - available)
        if missing:
            LOGGER.warning("Requested indices not found in CSV and will be ignored: %s", ", ".join(missing))
        before = len(df)
        df = df[df[cfg.index_col].isin(allowed)].copy()
        LOGGER.info("Filtered indices: %d -> %d rows", before, len(df))

    if cfg.min_px_count > 0:
        before = len(df)
        df = df[df[cfg.px_count_col].fillna(0) >= cfg.min_px_count].copy()
        LOGGER.info("Applied min_px_count=%d: %d -> %d rows", cfg.min_px_count, before, len(df))

    if cfg.max_cloud_scene is not None:
        before = len(df)
        df = df[df[cfg.cloud_col].fillna(np.inf) <= cfg.max_cloud_scene].copy()
        LOGGER.info(
            "Applied max_cloud_scene=%.3f: %d -> %d rows",
            cfg.max_cloud_scene,
            before,
            len(df),
        )

    if df.empty:
        raise ValueError("No valid rows remaining after filtering.")

    return df


def _aggregate_long_dataframe(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    df_work = df.copy()
    if cfg.time_grid_frequency:
        n_dates_before = int(pd.to_datetime(df_work[cfg.date_col]).nunique())
        df_work["_date_bin"] = _bin_dates_to_frequency(
            pd.to_datetime(df_work[cfg.date_col]),
            cfg.time_grid_frequency,
        )
        n_dates_after = int(df_work["_date_bin"].nunique())
        LOGGER.info(
            "Applied temporal binning freq=%s: unique dates %d -> %d",
            cfg.time_grid_frequency,
            n_dates_before,
            n_dates_after,
        )
        group_date_col = "_date_bin"
    else:
        group_date_col = cfg.date_col

    group_cols = [cfg.parcel_id_col, group_date_col, cfg.index_col]
    agg_spec = {
        cfg.value_col: "mean",
        cfg.cloud_col: "mean",
        cfg.px_count_col: "mean",
        cfg.tile_col: _mode_or_first,
        cfg.label_col: _mode_or_first,
    }
    if cfg.label_group_col in df.columns:
        agg_spec[cfg.label_group_col] = _mode_or_first

    aggregated = (
        df_work.groupby(group_cols, dropna=False, as_index=False)
        .agg(agg_spec)
        .sort_values([cfg.parcel_id_col, group_date_col, cfg.index_col])
    )
    if group_date_col != cfg.date_col:
        aggregated = aggregated.rename(columns={group_date_col: cfg.date_col})

    if aggregated.empty:
        raise ValueError("No rows left after grouping by parcel/date/index.")
    return aggregated


def _build_time_grid(date_values: pd.Series, cfg: DataConfig) -> pd.DatetimeIndex:
    min_date = pd.to_datetime(date_values.min())
    max_date = pd.to_datetime(date_values.max())
    if cfg.time_grid_frequency:
        grid = pd.date_range(start=min_date, end=max_date, freq=cfg.time_grid_frequency)
    else:
        grid = pd.DatetimeIndex(sorted(pd.to_datetime(date_values).unique()))
    if len(grid) == 0:
        raise ValueError("Temporal grid is empty.")
    return grid


def _stratify_target_if_possible(labels: np.ndarray, use_stratify: bool) -> Optional[np.ndarray]:
    if not use_stratify:
        return None
    bincount = np.bincount(labels)
    if len(bincount) < 2:
        return None
    if bincount.min() < 2:
        return None
    return labels


def _split_by_parcel(labels: np.ndarray, cfg: DataConfig) -> dict[str, np.ndarray]:
    if not 0 < cfg.test_size < 1:
        raise ValueError("test_size must be in (0, 1).")
    if not 0 <= cfg.val_size < 1:
        raise ValueError("val_size must be in [0, 1).")
    if cfg.test_size + cfg.val_size >= 1:
        raise ValueError("test_size + val_size must be < 1.")

    indices = np.arange(labels.shape[0])
    stratify_all = _stratify_target_if_possible(labels, cfg.stratify)

    try:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            shuffle=True,
            stratify=stratify_all,
        )
    except ValueError:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            shuffle=True,
            stratify=None,
        )

    val_ratio_adjusted = cfg.val_size / (1.0 - cfg.test_size)
    if val_ratio_adjusted <= 0 or len(train_val_idx) < 2:
        return {"train": np.sort(train_val_idx), "val": np.array([], dtype=np.int64), "test": np.sort(test_idx)}

    n_val = int(np.ceil(len(train_val_idx) * val_ratio_adjusted))
    if n_val <= 0 or n_val >= len(train_val_idx):
        return {"train": np.sort(train_val_idx), "val": np.array([], dtype=np.int64), "test": np.sort(test_idx)}

    stratify_tv = _stratify_target_if_possible(labels[train_val_idx], cfg.stratify)
    try:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=n_val,
            random_state=cfg.random_state,
            shuffle=True,
            stratify=stratify_tv,
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=n_val,
            random_state=cfg.random_state,
            shuffle=True,
            stratify=None,
        )

    return {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}


def _split_by_tile(labels: np.ndarray, parcel_tiles: np.ndarray, cfg: DataConfig) -> dict[str, np.ndarray]:
    unique_tiles = np.unique(parcel_tiles)
    if len(unique_tiles) < 3:
        LOGGER.warning(
            "split_method='tile' requested but only %d unique tiles found. Falling back to parcel split.",
            len(unique_tiles),
        )
        return _split_by_parcel(labels, cfg)

    if cfg.test_size + cfg.val_size >= 1:
        raise ValueError("test_size + val_size must be < 1.")

    tile_values, tile_counts = np.unique(parcel_tiles, return_counts=True)
    total_count = int(tile_counts.sum())
    train_ratio = 1.0 - cfg.test_size - cfg.val_size
    val_ratio = cfg.val_size
    test_ratio = cfg.test_size
    need_val = val_ratio > 0

    split_ids = np.array([0, 1, 2] if need_val else [0, 2], dtype=np.int64)
    best_score = float("inf")
    best_assignment: Optional[np.ndarray] = None

    def evaluate_assignment(assignment: np.ndarray) -> float:
        train_count = int(tile_counts[assignment == 0].sum())
        val_count = int(tile_counts[assignment == 1].sum()) if need_val else 0
        test_count = int(tile_counts[assignment == 2].sum())

        if train_count == 0 or test_count == 0:
            return float("inf")
        if need_val and val_count == 0:
            return float("inf")

        train_frac = train_count / total_count
        val_frac = val_count / total_count if need_val else 0.0
        test_frac = test_count / total_count
        score = abs(train_frac - train_ratio) + abs(test_frac - test_ratio)
        if need_val:
            score += abs(val_frac - val_ratio)
        return float(score)

    if len(tile_values) <= 12:
        for candidate in product(split_ids.tolist(), repeat=len(tile_values)):
            assignment = np.asarray(candidate, dtype=np.int64)
            score = evaluate_assignment(assignment)
            if score < best_score:
                best_score = score
                best_assignment = assignment
    else:
        rng = np.random.default_rng(cfg.random_state)
        for _ in range(25_000):
            assignment = rng.choice(split_ids, size=len(tile_values), replace=True)
            score = evaluate_assignment(assignment)
            if score < best_score:
                best_score = score
                best_assignment = assignment

    if best_assignment is None or not np.isfinite(best_score):
        LOGGER.warning("Could not optimize tile split. Falling back to parcel split.")
        return _split_by_parcel(labels, cfg)

    train_tiles = tile_values[best_assignment == 0]
    val_tiles = tile_values[best_assignment == 1] if need_val else np.array([], dtype=tile_values.dtype)
    test_tiles = tile_values[best_assignment == 2]

    train_idx = np.where(np.isin(parcel_tiles, train_tiles))[0]
    val_idx = np.where(np.isin(parcel_tiles, val_tiles))[0]
    test_idx = np.where(np.isin(parcel_tiles, test_tiles))[0]

    if len(train_idx) == 0 or len(test_idx) == 0:
        LOGGER.warning("Spatial split produced an empty split. Falling back to parcel split.")
        return _split_by_parcel(labels, cfg)

    LOGGER.info(
        "Optimized tile split | train tiles=%s (%d) | val tiles=%s (%d) | test tiles=%s (%d) | score=%.4f",
        train_tiles.tolist(),
        len(train_idx),
        val_tiles.tolist(),
        len(val_idx),
        test_tiles.tolist(),
        len(test_idx),
        best_score,
    )

    return {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}


def _create_splits(labels: np.ndarray, parcel_tiles: np.ndarray, cfg: DataConfig) -> dict[str, np.ndarray]:
    if cfg.split_method == "tile":
        return _split_by_tile(labels, parcel_tiles, cfg)
    return _split_by_parcel(labels, cfg)


def _build_tensor_dataset(df: pd.DataFrame, cfg: DataConfig) -> PreparedDataset:
    feature_names = sorted(df[cfg.index_col].astype(str).unique().tolist())
    if not feature_names:
        raise ValueError("No index names found in column '%s'." % cfg.index_col)

    parcel_ids = np.array(sorted(df[cfg.parcel_id_col].astype(str).unique()))
    time_grid = _build_time_grid(df[cfg.date_col], cfg)

    pivot = df.pivot_table(
        index=[cfg.parcel_id_col, cfg.date_col],
        columns=cfg.index_col,
        values=cfg.value_col,
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=feature_names)

    quality = (
        df.groupby([cfg.parcel_id_col, cfg.date_col], as_index=True)
        .agg(
            {
                cfg.cloud_col: "mean",
                cfg.px_count_col: "mean",
            }
        )
    )

    full_index = pd.MultiIndex.from_product(
        [parcel_ids, time_grid],
        names=[cfg.parcel_id_col, cfg.date_col],
    )
    pivot = pivot.reindex(full_index)
    quality = quality.reindex(full_index)

    n_parcels = len(parcel_ids)
    seq_len = len(time_grid)
    num_features = len(feature_names)

    observed_mask = pivot.notna().any(axis=1).to_numpy(dtype=bool).reshape(n_parcels, seq_len)
    features = (
        pivot.fillna(cfg.fill_value).to_numpy(dtype=np.float32).reshape(n_parcels, seq_len, num_features)
    )
    cloud_scene = quality[cfg.cloud_col].to_numpy(dtype=np.float32).reshape(n_parcels, seq_len)
    px_count = quality[cfg.px_count_col].to_numpy(dtype=np.float32).reshape(n_parcels, seq_len)
    cloud_scene = np.nan_to_num(cloud_scene, nan=100.0, posinf=100.0, neginf=100.0)
    px_count = np.nan_to_num(px_count, nan=0.0, posinf=0.0, neginf=0.0)
    cloud_scene = np.clip(cloud_scene, a_min=0.0, a_max=100.0).astype(np.float32)
    px_count = np.clip(px_count, a_min=0.0, a_max=None).astype(np.float32)

    day_vector = np.asarray(time_grid.dayofyear, dtype=np.int16)
    day_of_year = np.tile(day_vector[None, :], (n_parcels, 1))

    metadata_spec = {cfg.label_col: _mode_or_first, cfg.tile_col: _mode_or_first}
    if cfg.label_group_col in df.columns:
        metadata_spec[cfg.label_group_col] = _mode_or_first

    parcel_metadata = (
        df.groupby(cfg.parcel_id_col, as_index=True)
        .agg(metadata_spec)
        .reindex(parcel_ids)
    )
    if parcel_metadata[cfg.label_col].isna().any():
        raise ValueError("Missing labels for some parcels after aggregation.")

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(parcel_metadata[cfg.label_col].astype(str).to_numpy())
    label_names = label_encoder.classes_.tolist()
    parcel_tiles = parcel_metadata[cfg.tile_col].fillna("unknown").astype(str).to_numpy()

    group_labels: Optional[np.ndarray] = None
    group_label_names: list[str] = []
    if cfg.label_group_col in parcel_metadata.columns:
        group_series = parcel_metadata[cfg.label_group_col].fillna("").astype(str).str.strip()
        valid_group_mask = group_series != ""
        if valid_group_mask.any():
            group_encoder = LabelEncoder()
            group_encoder.fit(group_series[valid_group_mask].to_numpy())
            group_labels = np.full((len(parcel_ids),), -100, dtype=np.int64)
            group_labels[valid_group_mask.to_numpy()] = group_encoder.transform(
                group_series[valid_group_mask].to_numpy()
            )
            group_label_names = group_encoder.classes_.tolist()
            LOGGER.info(
                "Detected %d/%d parcels with group labels in '%s' (%d groups).",
                int(valid_group_mask.sum()),
                int(len(valid_group_mask)),
                cfg.label_group_col,
                len(group_label_names),
            )
        else:
            LOGGER.info(
                "Column '%s' exists but has no non-empty values. Group task will be disabled.",
                cfg.label_group_col,
            )

    min_obs_mask = observed_mask.sum(axis=1) >= cfg.min_obs_per_parcel
    if min_obs_mask.sum() == 0:
        raise ValueError(
            "All parcels were removed by min_obs_per_parcel=%d. Lower this threshold."
            % cfg.min_obs_per_parcel
        )
    if min_obs_mask.sum() < len(min_obs_mask):
        LOGGER.info(
            "Removed %d parcels with fewer than %d observed dates.",
            int((~min_obs_mask).sum()),
            cfg.min_obs_per_parcel,
        )

    features = features[min_obs_mask]
    day_of_year = day_of_year[min_obs_mask]
    observed_mask = observed_mask[min_obs_mask]
    cloud_scene = cloud_scene[min_obs_mask]
    px_count = px_count[min_obs_mask]
    labels = labels[min_obs_mask]
    parcel_ids = parcel_ids[min_obs_mask]
    parcel_tiles = parcel_tiles[min_obs_mask]
    if group_labels is not None:
        group_labels = group_labels[min_obs_mask]

    splits = _create_splits(labels=labels, parcel_tiles=parcel_tiles, cfg=cfg)

    return PreparedDataset(
        features=features.astype(np.float32),
        day_of_year=day_of_year.astype(np.int16),
        observed_mask=observed_mask.astype(bool),
        cloud_scene=cloud_scene.astype(np.float32),
        px_count=px_count.astype(np.float32),
        labels=labels.astype(np.int64),
        parcel_ids=parcel_ids.astype(str),
        parcel_tiles=parcel_tiles.astype(str),
        feature_names=feature_names,
        label_names=label_names,
        group_labels=group_labels.astype(np.int64) if group_labels is not None else None,
        group_label_names=group_label_names,
        time_grid=np.asarray(time_grid.values, dtype="datetime64[ns]"),
        splits=splits,
    )


def prepare_dataset(cfg: DataConfig) -> PreparedDataset:
    if cfg.prepared_npz_path:
        npz_path = Path(cfg.prepared_npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Prepared dataset not found: {npz_path}")
        prepared = load_prepared_dataset(npz_path)
    else:
        df = load_long_dataframe(cfg)
        aggregated = _aggregate_long_dataframe(df, cfg)
        prepared = _build_tensor_dataset(aggregated, cfg)

    if cfg.save_prepared_npz_path:
        save_prepared_dataset(prepared, cfg.save_prepared_npz_path)

    return prepared


def standardize_prepared_features(
    prepared: PreparedDataset,
    train_indices: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    """
    Standardize features using only train split statistics.
    Missing dates are excluded from stats via observed_mask and reset to 0 after scaling.
    """
    if train_indices.size == 0:
        raise ValueError("Cannot standardize features with an empty train split.")

    train_features = prepared.features[train_indices]  # [N_train, T, F]
    train_observed = prepared.observed_mask[train_indices][..., None].astype(np.float32)  # [N_train, T, 1]

    counts = train_observed.sum(axis=(0, 1)).reshape(1).repeat(prepared.num_features)
    if np.any(counts <= 0):
        raise ValueError("Found a feature with zero observed samples in train split.")

    mean = (train_features * train_observed).sum(axis=(0, 1)) / np.clip(counts, a_min=1.0, a_max=None)
    var = (
        ((train_features - mean[None, None, :]) ** 2) * train_observed
    ).sum(axis=(0, 1)) / np.clip(counts, a_min=1.0, a_max=None)
    std = np.sqrt(np.clip(var, a_min=eps**2, a_max=None))

    scaled = (prepared.features - mean[None, None, :]) / std[None, None, :]
    scaled[~prepared.observed_mask] = 0.0
    prepared.features = scaled.astype(np.float32)

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "count_per_feature": counts.astype(np.int64),
    }


def build_dataloaders(
    prepared: PreparedDataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    train_sampler: Optional[Sampler] = None,
    train_augmentation: Optional[TemporalAugmentationConfig] = None,
) -> dict[str, DataLoader]:
    dataloaders: dict[str, DataLoader] = {}
    for split_name, split_indices in prepared.splits.items():
        split_dataset = ParcelTimeSeriesDataset(
            prepared,
            split_indices,
            augmentation=train_augmentation if split_name == "train" else None,
        )
        use_sampler = split_name == "train" and train_sampler is not None
        dataloaders[split_name] = DataLoader(
            split_dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train" and not use_sampler),
            sampler=train_sampler if use_sampler else None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    return dataloaders


def save_prepared_dataset(prepared: PreparedDataset, path: str | Path) -> None:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    splits_json = json.dumps({k: v.tolist() for k, v in prepared.splits.items()})
    has_group_labels = prepared.group_labels is not None and len(prepared.group_label_names) > 0
    cloud_scene = (
        prepared.cloud_scene.astype(np.float32)
        if prepared.cloud_scene is not None
        else np.full(prepared.observed_mask.shape, 100.0, dtype=np.float32)
    )
    px_count = (
        prepared.px_count.astype(np.float32)
        if prepared.px_count is not None
        else np.zeros(prepared.observed_mask.shape, dtype=np.float32)
    )
    np.savez_compressed(
        save_path,
        features=prepared.features,
        day_of_year=prepared.day_of_year,
        observed_mask=prepared.observed_mask,
        cloud_scene=cloud_scene,
        px_count=px_count,
        labels=prepared.labels,
        group_labels=prepared.group_labels if has_group_labels else np.array([], dtype=np.int64),
        parcel_ids=prepared.parcel_ids,
        parcel_tiles=prepared.parcel_tiles,
        feature_names=np.asarray(prepared.feature_names, dtype=object),
        label_names=np.asarray(prepared.label_names, dtype=object),
        group_label_names=np.asarray(prepared.group_label_names, dtype=object),
        time_grid=prepared.time_grid.astype("datetime64[ns]").astype("int64"),
        splits_json=np.asarray([splits_json], dtype=object),
        has_group_labels=np.asarray([has_group_labels], dtype=np.bool_),
    )
    LOGGER.info("Saved prepared dataset to %s", save_path)


def load_prepared_dataset(path: str | Path) -> PreparedDataset:
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"NPZ dataset not found: {load_path}")

    with np.load(load_path, allow_pickle=True) as data:
        splits_json = str(data["splits_json"][0])
        splits = {k: np.asarray(v, dtype=np.int64) for k, v in json.loads(splits_json).items()}
        has_group_labels = bool(data["has_group_labels"][0]) if "has_group_labels" in data.files else False
        if has_group_labels and "group_labels" in data.files and "group_label_names" in data.files:
            group_labels = data["group_labels"].astype(np.int64)
            group_label_names = [str(x) for x in data["group_label_names"].tolist()]
            if group_labels.shape[0] == data["labels"].shape[0] and len(group_label_names) > 0:
                parsed_group_labels: Optional[np.ndarray] = group_labels
                parsed_group_label_names = group_label_names
            else:
                parsed_group_labels = None
                parsed_group_label_names = []
        else:
            parsed_group_labels = None
            parsed_group_label_names = []

        prepared = PreparedDataset(
            features=data["features"].astype(np.float32),
            day_of_year=data["day_of_year"].astype(np.int16),
            observed_mask=data["observed_mask"].astype(bool),
            cloud_scene=(
                data["cloud_scene"].astype(np.float32)
                if "cloud_scene" in data.files
                else np.full(data["observed_mask"].shape, 100.0, dtype=np.float32)
            ),
            px_count=(
                data["px_count"].astype(np.float32)
                if "px_count" in data.files
                else np.zeros(data["observed_mask"].shape, dtype=np.float32)
            ),
            labels=data["labels"].astype(np.int64),
            parcel_ids=data["parcel_ids"].astype(str),
            parcel_tiles=data["parcel_tiles"].astype(str),
            feature_names=[str(x) for x in data["feature_names"].tolist()],
            label_names=[str(x) for x in data["label_names"].tolist()],
            group_labels=parsed_group_labels,
            group_label_names=parsed_group_label_names,
            time_grid=data["time_grid"].astype("int64").astype("datetime64[ns]"),
            splits=splits,
        )
    return prepared
