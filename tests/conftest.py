from pathlib import Path

import pytest

from src.config import PipelineConfig


@pytest.fixture
def small_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        random_seed=7,
        n_days=70,
        n_products=2,
        n_categories=2,
        n_stores=2,
        n_estimators=6,
        forecast_horizon=4,
        output_dir=str(tmp_path / "outputs"),
    )
