"""Cleaning, calendar features, and supervised-learning feature matrices."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

KEY_COLUMNS = ["product_id", "category", "store_id"]
REQUIRED_COLUMNS = [*KEY_COLUMNS, "date", "units_sold"]
CATEGORICAL_COLUMNS = KEY_COLUMNS


def clean_sales_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize raw sales data.

    Invalid dates and rows without business keys are removed. Duplicate
    product-store-date observations are aggregated so lag calculations are
    well-defined.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"sales data is missing required columns: {missing}")
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["units_sold"] = pd.to_numeric(frame["units_sold"], errors="coerce")
    frame = frame.dropna(subset=["date", *KEY_COLUMNS, "units_sold"])
    frame["units_sold"] = frame["units_sold"].clip(lower=0).round().astype(int)
    for column in ("promotion", "stockout"):
        if column not in frame:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).clip(0, 1).round().astype(int)

    for column in frame.columns:
        if column not in {*KEY_COLUMNS, "date", "promotion", "stockout", "units_sold"}:
            try:
                frame[column] = pd.to_numeric(frame[column])
            except (TypeError, ValueError):
                pass
    aggregations: dict[str, str] = {"units_sold": "sum", "promotion": "max", "stockout": "max"}
    for column in frame.columns:
        if column not in {*KEY_COLUMNS, "date", *aggregations}:
            aggregations[column] = "mean" if pd.api.types.is_numeric_dtype(frame[column]) else "first"
    return (
        frame.groupby([*KEY_COLUMNS, "date"], as_index=False, sort=False)
        .agg(aggregations)
        .sort_values([*KEY_COLUMNS, "date"])
        .reset_index(drop=True)
    )


def build_features(data: pd.DataFrame, drop_incomplete: bool = True) -> pd.DataFrame:
    """Add prior-demand lags, rolling statistics, and calendar features.

    Rolling values use only prior observations (``shift(1)``), preventing the
    current target from leaking into its own feature.
    """
    frame = clean_sales_data(data) if not {"lag_1", "lag_7"}.issubset(data.columns) else data.copy()
    frame = frame.sort_values([*KEY_COLUMNS, "date"]).reset_index(drop=True)
    grouped = frame.groupby(KEY_COLUMNS, sort=False)["units_sold"]
    frame["lag_1"] = grouped.shift(1)
    frame["lag_7"] = grouped.shift(7)
    frame["rolling_mean_7"] = grouped.transform(lambda values: values.shift(1).rolling(7, min_periods=7).mean())
    frame["rolling_std_7"] = grouped.transform(lambda values: values.shift(1).rolling(7, min_periods=7).std())
    frame["rolling_mean_28"] = grouped.transform(lambda values: values.shift(1).rolling(28, min_periods=28).mean())
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["day_of_month"] = frame["date"].dt.day
    frame["month"] = frame["date"].dt.month
    frame["day_of_year"] = frame["date"].dt.dayofyear
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)
    frame["trend_day"] = (frame["date"] - frame["date"].min()).dt.days
    if drop_incomplete:
        required = ["lag_1", "lag_7", "rolling_mean_7", "rolling_std_7", "rolling_mean_28"]
        frame = frame.dropna(subset=required).reset_index(drop=True)
    return frame


def encode_features(feature_frame: pd.DataFrame, feature_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Convert feature rows into a dense, training-aligned model matrix."""
    excluded = {"date", "units_sold", "stockout", "true_demand", "unit_price"}
    columns = [column for column in feature_frame.columns if column not in excluded]
    matrix = pd.get_dummies(feature_frame[columns], columns=CATEGORICAL_COLUMNS, dtype=float)
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if feature_columns is not None:
        matrix = matrix.reindex(columns=list(feature_columns), fill_value=0.0)
    return matrix.astype(float)
