"""Reorder-point and safety-stock calculations."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .preprocessing.features import KEY_COLUMNS, clean_sales_data


def normal_z_value(service_level: float) -> float:
    """Return the standard-normal quantile without requiring scipy."""
    if not 0 < service_level < 1:
        raise ValueError("service_level must be between 0 and 1")
    return float(NormalDist().inv_cdf(service_level))


def inventory_status(current_inventory: float, reorder_point: float, target_stock: float) -> str:
    """Classify a location's inventory position in business language."""
    if current_inventory < reorder_point:
        return "Reorder now"
    if current_inventory < target_stock:
        return "Monitor"
    return "Healthy"


def calculate_inventory_plan(
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    config: PipelineConfig | None = None,
    inventory_state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate safety stock, reorder point, and order recommendations.

    If no on-hand snapshot is provided, current inventory is a transparent
    estimate: recent seven-day average demand multiplied by a configurable
    number of cover days.  Supplying ``inventory_state`` with the key columns
    and ``on_hand_units`` replaces this estimate.
    """
    config = config or PipelineConfig()
    required_forecast = {*KEY_COLUMNS, "forecast_units"}
    if not required_forecast.issubset(forecast.columns):
        raise ValueError(f"forecast is missing required columns: {sorted(required_forecast - set(forecast.columns))}")
    clean = clean_sales_data(history)
    if clean.empty:
        raise ValueError("history cannot be empty")
    forecast_grouped = forecast.groupby(KEY_COLUMNS, as_index=False)["forecast_units"].mean()
    history_grouped = clean.groupby(KEY_COLUMNS, as_index=False).agg(
        recent_mean_units=("units_sold", lambda values: float(values.tail(7).mean())),
        daily_std_units=("units_sold", lambda values: float(values.std(ddof=1)) if len(values) > 1 else 0.0),
        historical_stockout_rate=("stockout", "mean"),
    )
    plan = forecast_grouped.merge(history_grouped, on=KEY_COLUMNS, how="left")
    if inventory_state is not None:
        state = inventory_state[[*KEY_COLUMNS, "on_hand_units"]].copy()
        state["on_hand_units"] = pd.to_numeric(state["on_hand_units"], errors="coerce")
        plan = plan.merge(state, on=KEY_COLUMNS, how="left")
    else:
        plan["on_hand_units"] = np.nan
    plan["current_inventory"] = plan["on_hand_units"].fillna(
        plan["recent_mean_units"] * config.inventory_cover_days
    ).clip(lower=0)

    z = normal_z_value(config.service_level)
    plan["forecast_daily_demand"] = plan["forecast_units"].clip(lower=0)
    plan["demand_during_lead_time"] = plan["forecast_daily_demand"] * config.lead_time_days
    plan["safety_stock"] = z * plan["daily_std_units"].fillna(0).clip(lower=0) * np.sqrt(config.lead_time_days)
    plan["reorder_point"] = plan["demand_during_lead_time"] + plan["safety_stock"]
    plan["target_stock"] = plan["forecast_daily_demand"] * (config.lead_time_days + config.review_period_days) + plan["safety_stock"]
    plan["recommended_order_qty"] = np.ceil(
        (plan["target_stock"] - plan["current_inventory"]).clip(lower=0)
    ).astype(int)
    plan["inventory_status"] = [
        inventory_status(current, reorder, target)
        for current, reorder, target in zip(
            plan["current_inventory"], plan["reorder_point"], plan["target_stock"]
        )
    ]
    columns = [
        *KEY_COLUMNS,
        "forecast_daily_demand",
        "recent_mean_units",
        "daily_std_units",
        "historical_stockout_rate",
        "current_inventory",
        "demand_during_lead_time",
        "safety_stock",
        "reorder_point",
        "target_stock",
        "recommended_order_qty",
        "inventory_status",
    ]
    return plan[columns].sort_values(["inventory_status", *KEY_COLUMNS]).reset_index(drop=True)
