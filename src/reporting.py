"""CSV, JSON, and PNG reporting for the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_business_insights(
    forecast: pd.DataFrame,
    inventory_plan: pd.DataFrame,
    metrics: dict[str, Any],
) -> pd.DataFrame:
    """Turn model and inventory outputs into concise, readable insights."""
    total_order = int(inventory_plan["recommended_order_qty"].sum()) if not inventory_plan.empty else 0
    reorder_count = int((inventory_plan["inventory_status"] == "Reorder now").sum()) if not inventory_plan.empty else 0
    peak = forecast.groupby("date", as_index=False)["forecast_units"].sum() if not forecast.empty else pd.DataFrame()
    peak_date = peak.loc[peak["forecast_units"].idxmax(), "date"] if not peak.empty else None
    rows = [
        {"insight_type": "Model quality", "insight": f"Time-aware holdout MAE is {float(metrics.get('mae', 0)):.2f} units.", "value": float(metrics.get("mae", 0))},
        {"insight_type": "Demand outlook", "insight": f"The forecast peaks on {peak_date:%Y-%m-%d} across all locations." if peak_date is not None else "No forecast rows were produced.", "value": float(peak["forecast_units"].max()) if not peak.empty else 0.0},
        {"insight_type": "Inventory action", "insight": f"{reorder_count} location(s) are below reorder point; recommended replenishment is {total_order:,} units.", "value": total_order},
        {"insight_type": "Stockout signal", "insight": f"Historical stockout rate averages {inventory_plan['historical_stockout_rate'].mean() * 100:.1f}% across planned locations." if not inventory_plan.empty else "No inventory locations were produced.", "value": float(inventory_plan["historical_stockout_rate"].mean()) if not inventory_plan.empty else 0.0},
    ]
    return pd.DataFrame(rows)


def _save_forecast_plot(history: pd.DataFrame, forecast: pd.DataFrame, path: Path) -> None:
    actual = history.groupby("date", as_index=False)["units_sold"].sum()
    predicted = forecast.groupby("date", as_index=False)["forecast_units"].sum()
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(actual["date"], actual["units_sold"], label="Observed sales", color="#2563eb", linewidth=1.5)
    axis.plot(predicted["date"], predicted["forecast_units"], label="Forecast", color="#dc2626", marker="o", markersize=3)
    axis.set_title("Total retail demand: observed history and forecast")
    axis.set_xlabel("Date")
    axis.set_ylabel("Units")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _save_inventory_plot(inventory_plan: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(7, 4))
    counts = inventory_plan["inventory_status"].value_counts().reindex(["Reorder now", "Monitor", "Healthy"], fill_value=0)
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=["#dc2626", "#f59e0b", "#16a34a"], legend=False, ax=axis)
    axis.set_title("Inventory status by product and store")
    axis.set_xlabel("")
    axis.set_ylabel("Locations")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def save_pipeline_outputs(
    output_dir: str | Path,
    clean_sales: pd.DataFrame,
    features: pd.DataFrame,
    forecast: pd.DataFrame,
    inventory_plan: pd.DataFrame,
    insights: pd.DataFrame,
    metrics: dict[str, Any],
) -> dict[str, Path]:
    """Persist tables, metrics, and plots; return the exact written paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "clean_sales.csv": clean_sales,
        "features.csv": features,
        "forecast.csv": forecast,
        "inventory_plan.csv": inventory_plan,
        "business_insights.csv": insights,
    }
    paths: dict[str, Path] = {}
    for filename, table in tables.items():
        path = directory / filename
        table.to_csv(path, index=False)
        paths[filename] = path
    metrics_path = directory / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    paths["metrics.json"] = metrics_path
    forecast_plot = directory / "forecast_overview.png"
    inventory_plot = directory / "inventory_status.png"
    _save_forecast_plot(clean_sales, forecast, forecast_plot)
    _save_inventory_plot(inventory_plan, inventory_plot)
    paths[forecast_plot.name] = forecast_plot
    paths[inventory_plot.name] = inventory_plot
    return paths
