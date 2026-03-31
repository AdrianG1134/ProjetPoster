from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

SplitMethod = Literal["parcel", "tile"]
PoolingType = Literal["cls", "mean"]
SchedulerType = Literal["none", "plateau", "cosine"]
LossType = Literal["cross_entropy", "focal", "balanced_softmax", "logit_adjusted"]


@dataclass
class DataConfig:
    csv_path: str = (
        "data/s2_herault_2024_full_year_5day_cloudmask_fast/"
        "indices_parcelles_2024-01-01_2024-12-31_win5d_with_labels.csv"
    )
    output_dir: str = "outputs_transformer"
    parcel_id_col: str = "ID_PARCEL"
    date_col: str = "date"
    index_col: str = "index"
    value_col: str = "value_mean"
    label_col: str = "label"
    label_group_col: str = "label_group"
    tile_col: str = "tile"
    cloud_col: str = "cloud_scene"
    px_count_col: str = "px_count"
    index_filter: list[str] = field(default_factory=list)
    min_px_count: int = 0
    max_cloud_scene: Optional[float] = None
    min_obs_per_parcel: int = 4
    fill_value: float = 0.0
    time_grid_frequency: Optional[str] = None
    split_method: SplitMethod = "parcel"
    stratify: bool = True
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    prepared_npz_path: Optional[str] = None
    save_prepared_npz_path: Optional[str] = None


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    pooling: PoolingType = "cls"
    use_layer_norm_first: bool = True


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    standardize_features: bool = True
    standardize_eps: float = 1e-6
    loss_type: LossType = "focal"
    focal_gamma: float = 2.0
    logit_adjust_tau: float = 1.0
    use_group_task: bool = False
    group_loss_weight: float = 0.3
    class_weighting: bool = True
    class_weight_power: float = 0.5
    weighted_sampler: bool = False
    sampler_power: float = 1.0
    temporal_augmentation: bool = False
    time_mask_ratio: float = 0.0
    jitter_std: float = 0.0
    phase2_rare_finetune: bool = False
    phase2_epochs: int = 12
    phase2_learning_rate: float = 1e-4
    phase2_sampler_power: float = 1.0
    phase2_rare_quantile: float = 0.25
    phase2_rare_count_threshold: Optional[int] = None
    phase2_rare_boost: float = 2.0
    phase2_early_stopping_patience: int = 6
    scheduler: SchedulerType = "plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-6
    early_stopping_patience: int = 12
    gradient_clip_norm: Optional[float] = 1.0
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()
