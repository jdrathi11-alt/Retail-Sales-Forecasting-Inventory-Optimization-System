"""Transparent time-aware demand forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import PipelineConfig
from .preprocessing.features import KEY_COLUMNS, build_features, clean_sales_data, encode_features


@dataclass
class ForecastModel:
    """Fitted model plus the metadata needed for reproducible inference."""

    model: RandomForestRegressor
    feature_columns: list[str]
    metrics: dict[str, float | int | str]
    cutoff_date: pd.Timestamp


def _safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    nonzero = actual != 0
    if not np.any(nonzero):
        return 0.0
    return float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)


def _as_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    return data if {"lag_1", "lag_7", "rolling_mean_28"}.issubset(data.columns) else build_features(data)


def train_forecaster(data: pd.DataFrame, config: PipelineConfig | None = None) -> ForecastModel:
    """Fit a deterministic random forest using a chronological holdout.

    The final fraction of calendar dates is held out, so rows from the future
    never influence training.  Random forest is used here because it is robust
    for mixed calendar/categorical features and easy to inspect via feature
    importances.
    """
    config = config or PipelineConfig()
    frame = _as_feature_frame(data).sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise ValueError("no complete feature rows are available for training")
    unique_dates = pd.Index(frame["date"].drop_duplicates().sort_values())
    split_index = max(1, int(len(unique_dates) * (1 - config.test_fraction)))
    split_index = min(split_index, len(unique_dates) - 1)
    cutoff_date = pd.Timestamp(unique_dates[split_index])
    train = frame[frame["date"] < cutoff_date].copy()
    test = frame[frame["date"] >= cutoff_date].copy()
    if train.empty or test.empty:
        raise ValueError("time-aware split produced an empty train or test set")

    x_train = encode_features(train)
    x_test = encode_features(test, x_train.columns)
    y_train = train["units_sold"].to_numpy(dtype=float)
    y_test = test["units_sold"].to_numpy(dtype=float)
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_seed,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    predictions = np.maximum(0.0, model.predict(x_test))
    metrics: dict[str, float | int | str] = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mape_percent": _safe_mape(y_test, predictions),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_start": cutoff_date.strftime("%Y-%m-%d"),
    }
    return ForecastModel(model, list(x_train.columns), metrics, cutoff_date)


def forecast_future(
    fitted: ForecastModel,
    history: pd.DataFrame,
    horizon: int,
    promotion_plan: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forecast the next ``horizon`` days recursively for every key.

    Lag and rolling features for day ``t+2`` include the prediction made for
    day ``t+1``.  A promotion plan can provide known future promotions; absent
    a plan, future promotions default to zero (a conservative baseline).
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    clean = clean_sales_data(history).sort_values([*KEY_COLUMNS, "date"])
    if clean.empty:
        raise ValueError("history cannot be empty")
    plan = None
    if promotion_plan is not None:
        plan = promotion_plan.copy()
        if "date" not in plan or not set(KEY_COLUMNS).issubset(plan.columns):
            raise ValueError("promotion_plan needs date, product_id, category, and store_id")
        plan["date"] = pd.to_datetime(plan["date"], errors="coerce")
        plan["promotion"] = pd.to_numeric(plan.get("promotion", 0), errors="coerce").fillna(0).clip(0, 1).round().astype(int)

    last_date = pd.Timestamp(clean["date"].max())
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    rows: list[dict[str, object]] = []
    for keys, group in clean.groupby(KEY_COLUMNS, sort=True):
        product_id, category, store_id = keys
        demand_history = group["units_sold"].astype(float).tolist()
        first_date = pd.Timestamp(group["date"].min())
        for date in future_dates:
            key_values = {
                "product_id": product_id,
                "category": category,
                "store_id": store_id,
            }
            promotion = 0
            if plan is not None:
                matches = plan[(plan["date"] == date)]
                for key, value in key_values.items():
                    matches = matches[matches[key] == value]
                if not matches.empty:
                    promotion = int(matches.iloc[0]["promotion"])
            recent_7 = np.asarray(demand_history[-7:], dtype=float)
            recent_28 = np.asarray(demand_history[-28:], dtype=float)
            feature_row = {
                **key_values,
                "date": date,
                "promotion": promotion,
                "lag_1": demand_history[-1],
                "lag_7": demand_history[-7],
                "rolling_mean_7": float(np.mean(recent_7)),
                "rolling_std_7": float(np.std(recent_7, ddof=1)) if len(recent_7) > 1 else 0.0,
                "rolling_mean_28": float(np.mean(recent_28)),
                "day_of_week": date.dayofweek,
                "day_of_month": date.day,
                "month": date.month,
                "day_of_year": date.dayofyear,
                "is_weekend": int(date.dayofweek in (5, 6)),
                "trend_day": (date - first_date).days,
            }
            encoded = encode_features(pd.DataFrame([feature_row]), fitted.feature_columns)
            prediction = float(max(0.0, fitted.model.predict(encoded)[0]))
            rows.append({**key_values, "date": date, "promotion": promotion, "forecast_units": prediction})
            demand_history.append(prediction)
    return pd.DataFrame(rows).sort_values([*KEY_COLUMNS, "date"]).reset_index(drop=True)


def top_feature_importance(fitted: ForecastModel, limit: int = 10) -> pd.DataFrame:
    """Return the most influential model features for a simple explanation."""
    values = pd.DataFrame({"feature": fitted.feature_columns, "importance": fitted.model.feature_importances_})
    return values.sort_values("importance", ascending=False).head(limit).reset_index(drop=True)
