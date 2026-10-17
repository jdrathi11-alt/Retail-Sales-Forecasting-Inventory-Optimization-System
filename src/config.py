"""Configuration for the deterministic synthetic retail pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Small, reproducible defaults suitable for a local demo or tutorial."""

    random_seed: int = 42
    start_date: str = "2023-01-01"
    n_days: int = 365
    n_products: int = 8
    n_categories: int = 3
    n_stores: int = 4
    include_stockout_effects: bool = True
    stockout_probability: float = 0.035
    promotion_probability: float = 0.12
    forecast_horizon: int = 14
    test_fraction: float = 0.20
    n_estimators: int = 80
    max_depth: int | None = 12
    min_samples_leaf: int = 2
    lead_time_days: int = 7
    review_period_days: int = 7
    service_level: float = 0.95
    inventory_cover_days: int = 7
    output_dir: str = "outputs"

    def __post_init__(self) -> None:
        """Fail early on settings that would make the pipeline misleading."""
        if self.n_days < 35:
            raise ValueError("n_days must be at least 35 for lag and rolling features")
        if self.forecast_horizon < 1:
            raise ValueError("forecast_horizon must be positive")
        if not 0 < self.test_fraction < 0.5:
            raise ValueError("test_fraction must be between 0 and 0.5")
        if not 0 < self.service_level < 1:
            raise ValueError("service_level must be between 0 and 1")
        if not 0 <= self.stockout_probability <= 1:
            raise ValueError("stockout_probability must be between 0 and 1")
        if not 0 <= self.promotion_probability <= 1:
            raise ValueError("promotion_probability must be between 0 and 1")

    @property
    def output_path(self) -> Path:
        """Return the configured output directory as a ``Path``."""
        return Path(self.output_dir)
