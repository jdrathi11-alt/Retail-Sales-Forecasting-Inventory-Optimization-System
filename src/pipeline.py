"""End-to-end orchestration for simulation, forecasting, and replenishment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelineConfig
from .data_simulation import simulate_sales_data
from .forecasting import ForecastModel, forecast_future, train_forecaster, top_feature_importance
from .inventory import calculate_inventory_plan
from .preprocessing.features import build_features, clean_sales_data
from .reporting import generate_business_insights, save_pipeline_outputs


@dataclass
class PipelineResult:
    """All useful in-memory artifacts from one pipeline run."""

    raw_sales: pd.DataFrame
    clean_sales: pd.DataFrame
    features: pd.DataFrame
    model: ForecastModel
    forecast: pd.DataFrame
    inventory_plan: pd.DataFrame
    insights: pd.DataFrame
    output_paths: dict[str, Path]


def run_pipeline(config: PipelineConfig | None = None, output_dir: str | Path | None = None) -> PipelineResult:
    """Run every stage and save user-facing outputs."""
    config = config or PipelineConfig()
    if output_dir is not None:
        config = replace(config, output_dir=str(output_dir))
    raw_sales = simulate_sales_data(config)
    clean_sales = clean_sales_data(raw_sales)
    features = build_features(clean_sales)
    model = train_forecaster(features, config)
    forecast = forecast_future(model, clean_sales, config.forecast_horizon)
    inventory_plan = calculate_inventory_plan(forecast, clean_sales, config)
    metrics: dict[str, Any] = dict(model.metrics)
    importance = top_feature_importance(model)
    metrics["top_features"] = importance.to_dict(orient="records")
    insights = generate_business_insights(forecast, inventory_plan, metrics)
    output_paths = save_pipeline_outputs(
        config.output_path,
        clean_sales,
        features,
        forecast,
        inventory_plan,
        insights,
        metrics,
    )
    return PipelineResult(raw_sales, clean_sales, features, model, forecast, inventory_plan, insights, output_paths)
