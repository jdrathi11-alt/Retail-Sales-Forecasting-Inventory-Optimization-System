"""Streamlit portfolio dashboard for the retail forecasting pipeline.

The dashboard deliberately treats the CSV and JSON artifacts produced by
``src.pipeline.run_pipeline`` as its data contract.  That keeps the UI useful
for both a freshly generated demo and a previously saved pipeline run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:  # Keep helper functions importable in environments that only run checks.
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI guard.
    st = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
REQUIRED_OUTPUTS = {
    "clean_sales.csv": "clean_sales",
    "forecast.csv": "forecast",
    "inventory_plan.csv": "inventory_plan",
    "metrics.json": "metrics",
}


def _ensure_repo_on_path() -> None:
    """Allow ``streamlit run app/streamlit_app.py`` from the repository root."""

    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_pipeline_outputs(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> tuple[dict[str, Any], list[str]]:
    """Load saved pipeline artifacts and return ``(artifacts, errors)``.

    Missing artifacts are reported as friendly messages instead of raising so
    that the dashboard can guide a first-time user to generate demo data.
    """

    directory = Path(output_dir)
    artifacts: dict[str, Any] = {}
    errors: list[str] = []
    for filename, key in REQUIRED_OUTPUTS.items():
        path = directory / filename
        if not path.exists():
            errors.append(f"Missing {filename}")
            continue
        try:
            if path.suffix == ".json":
                artifacts[key] = json.loads(path.read_text(encoding="utf-8"))
            else:
                artifacts[key] = pd.read_csv(path, parse_dates=["date"] if key in {"clean_sales", "forecast"} else None)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"Could not read {filename}: {exc}")

    insights_path = directory / "business_insights.csv"
    if insights_path.exists():
        try:
            artifacts["insights"] = pd.read_csv(insights_path)
        except (OSError, ValueError) as exc:
            errors.append(f"Could not read business_insights.csv: {exc}")
    return artifacts, errors


def generate_demo_outputs(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    n_days: int = 365,
    forecast_horizon: int = 14,
    n_products: int = 8,
    n_stores: int = 4,
    random_seed: int = 42,
) -> Any:
    """Run the existing pipeline with transparent, dashboard-friendly defaults."""

    _ensure_repo_on_path()
    from src.config import PipelineConfig
    from src.pipeline import run_pipeline

    config = PipelineConfig(
        random_seed=random_seed,
        n_days=n_days,
        n_products=n_products,
        n_categories=min(3, n_products),
        n_stores=n_stores,
        forecast_horizon=forecast_horizon,
        output_dir=str(output_dir),
    )
    return run_pipeline(config)


def _number(value: Any, decimals: int = 0) -> str:
    """Format a dashboard number while keeping missing values readable."""

    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _demand_series(clean_sales: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Aggregate history and forecast into a chart-ready date series."""

    actual = clean_sales.groupby("date", as_index=False)["units_sold"].sum().rename(columns={"units_sold": "Observed sales"})
    predicted = forecast.groupby("date", as_index=False)["forecast_units"].sum().rename(columns={"forecast_units": "Forecast"})
    return actual.merge(predicted, on="date", how="outer").sort_values("date").set_index("date")


def _make_demand_chart(clean_sales: pd.DataFrame, forecast: pd.DataFrame) -> Any:
    """Create a compact, labeled history-versus-forecast chart."""

    import matplotlib.pyplot as plt

    series = _demand_series(clean_sales, forecast)
    figure, axis = plt.subplots(figsize=(11, 4.6))
    axis.plot(series.index, series["Observed sales"], label="Observed sales", color="#2563eb", linewidth=1.8)
    axis.plot(series.index, series["Forecast"], label="Forecast", color="#d97706", linewidth=2.2, marker="o", markersize=3)
    if not forecast.empty:
        axis.axvline(forecast["date"].min(), color="#64748b", linestyle="--", linewidth=1, label="Forecast starts")
    axis.set_title("Daily demand across all products and stores", loc="left", fontweight="bold")
    axis.set_xlabel("Date")
    axis.set_ylabel("Units")
    axis.grid(axis="y", alpha=0.25)
    axis.grid(axis="x", visible=False)
    axis.legend(frameon=False, ncols=3, loc="upper left")
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def _make_status_chart(inventory_plan: pd.DataFrame) -> Any:
    """Create a status-count chart with labels that do not rely on color alone."""

    import matplotlib.pyplot as plt

    order = ["Reorder now", "Monitor", "Healthy"]
    counts = inventory_plan["inventory_status"].value_counts().reindex(order, fill_value=0)
    colors = ["#b91c1c", "#b45309", "#15803d"]
    figure, axis = plt.subplots(figsize=(7, 4.6))
    bars = axis.bar(counts.index, counts.values, color=colors)
    axis.set_title("Inventory health by location", loc="left", fontweight="bold")
    axis.set_ylabel("Product-store locations")
    axis.grid(axis="y", alpha=0.25)
    axis.grid(axis="x", visible=False)
    for bar, value in zip(bars, counts.values):
        axis.annotate(str(int(value)), (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")
    figure.tight_layout()
    return figure


def _inventory_display(plan: pd.DataFrame) -> pd.DataFrame:
    """Return a readable subset of the inventory plan for the main table."""

    columns = {
        "product_id": "Product",
        "category": "Category",
        "store_id": "Store",
        "forecast_daily_demand": "Forecast / day",
        "current_inventory": "On hand",
        "reorder_point": "Reorder point",
        "target_stock": "Target stock",
        "recommended_order_qty": "Order quantity",
        "inventory_status": "Status",
        "historical_stockout_rate": "Historical stockout rate",
    }
    available = [column for column in columns if column in plan.columns]
    display = plan[available].rename(columns=columns).copy()
    for column in ["Forecast / day", "On hand", "Reorder point", "Target stock"]:
        if column in display:
            display[column] = display[column].round(1)
    if "Historical stockout rate" in display:
        display["Historical stockout rate"] = (display["Historical stockout rate"] * 100).round(1).map(lambda value: f"{value:.1f}%")
    return display


def _render_sidebar(artifacts: dict[str, Any], output_dir: Path) -> tuple[int, int, int, int]:
    """Render controls and return the selected demo-generation settings."""

    st.sidebar.header("Dashboard guide")
    st.sidebar.markdown(
        "**Start here:** review the demand outlook, then use the inventory table to decide which product-store locations need replenishment."
    )
    st.sidebar.markdown("**Forecast** estimates daily units for the next horizon. **Reorder point** combines lead-time demand and safety stock. **Order quantity** tops inventory up to the target stock.")
    st.sidebar.divider()
    st.sidebar.subheader("Generate demo data")
    st.sidebar.caption("No credentials or external data are required. The pipeline uses a seeded synthetic retail dataset.")
    n_days = st.sidebar.slider("Historical days", min_value=35, max_value=730, value=365, step=5, help="More days provide more history but take longer to fit.")
    horizon = st.sidebar.slider("Forecast horizon (days)", min_value=1, max_value=60, value=14, help="Number of future calendar days to forecast.")
    products = st.sidebar.slider("Products", min_value=2, max_value=12, value=8)
    stores = st.sidebar.slider("Stores", min_value=1, max_value=8, value=4)
    if st.sidebar.button("Generate demo data", type="primary", use_container_width=True):
        with st.spinner("Running simulation, forecast, and inventory planning…"):
            try:
                generate_demo_outputs(output_dir, n_days=n_days, forecast_horizon=horizon, n_products=products, n_stores=stores)
                st.sidebar.success(f"Saved fresh outputs to {output_dir.relative_to(REPO_ROOT) if output_dir.is_relative_to(REPO_ROOT) else output_dir}.")
                st.rerun()
            except Exception as exc:  # pragma: no cover - depends on runtime data/environment.
                st.sidebar.error(f"The demo run could not finish: {exc}")
    if artifacts:
        st.sidebar.caption(f"Loaded artifacts from `{output_dir}`.")
    return n_days, horizon, products, stores


def main() -> None:
    """Render the interactive dashboard."""

    if st is None:
        raise RuntimeError("Streamlit is not installed. Run `python -m pip install -r requirements.txt`, then start the app again.")

    _ensure_repo_on_path()
    st.set_page_config(page_title="Retail Forecasting Cockpit", page_icon="📦", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        [data-testid="stMetric"] { border: 1px solid #dbe3ec; border-radius: 0.5rem; padding: 0.85rem 1rem; background: #f8fafc; }
        [data-testid="stMetricLabel"] { color: #475569; }
        .dashboard-kicker { color: #2563eb; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    output_dir = DEFAULT_OUTPUT_DIR
    artifacts, errors = load_pipeline_outputs(output_dir)
    _render_sidebar(artifacts, output_dir)

    st.markdown('<div class="dashboard-kicker">Retail analytics portfolio project</div>', unsafe_allow_html=True)
    st.title("Demand forecast & inventory cockpit")
    st.write("A transparent view from simulated sales history to replenishment action. Use the controls in the sidebar to create a reproducible demo run.")

    if errors:
        st.info("**Data is not ready yet.** " + "; ".join(errors) + ". Use **Generate demo data** in the sidebar to create the dashboard artifacts.")
        st.stop()

    clean_sales = artifacts["clean_sales"]
    forecast = artifacts["forecast"]
    inventory_plan = artifacts["inventory_plan"]
    metrics = artifacts["metrics"]

    total_forecast = forecast["forecast_units"].sum() if not forecast.empty else 0
    total_order = inventory_plan["recommended_order_qty"].sum() if not inventory_plan.empty else 0
    reorder_count = int((inventory_plan["inventory_status"] == "Reorder now").sum()) if not inventory_plan.empty else 0
    mae = metrics.get("mae")
    forecast_start = pd.to_datetime(forecast["date"]).min() if not forecast.empty else None
    forecast_end = pd.to_datetime(forecast["date"]).max() if not forecast.empty else None

    metric_columns = st.columns(4)
    metric_columns[0].metric("Forecast units", _number(total_forecast), help="Total predicted units across every product-store location and forecast day.")
    metric_columns[1].metric("Suggested replenishment", _number(total_order), help="Units recommended to reach target stock across planned locations.")
    metric_columns[2].metric("Reorder alerts", f"{reorder_count:,}", help="Locations whose current inventory is below the calculated reorder point.")
    metric_columns[3].metric("Holdout MAE", f"{_number(mae, 1)} units", help="Mean absolute error on the chronological test period.")

    if forecast_start is not None and forecast_end is not None:
        st.caption(f"Forecast window: **{forecast_start:%d %b %Y} → {forecast_end:%d %b %Y}** · History through **{pd.to_datetime(clean_sales['date']).max():%d %b %Y}**")

    chart_column, status_column = st.columns([1.65, 1])
    with chart_column:
        st.subheader("Demand trend and forecast")
        st.pyplot(_make_demand_chart(clean_sales, forecast), use_container_width=True)
        st.caption("Observed sales are aggregated from the cleaned history. Forecast values begin after the final observed date.")
    with status_column:
        st.subheader("Inventory health")
        st.pyplot(_make_status_chart(inventory_plan), use_container_width=True)
        st.caption("Status labels are based on current inventory compared with reorder point and target stock.")

    st.divider()
    st.subheader("Reorder alerts")
    alerts = inventory_plan[inventory_plan["inventory_status"] == "Reorder now"].sort_values("recommended_order_qty", ascending=False)
    if alerts.empty:
        st.success("No locations are currently below their reorder point.")
    else:
        st.warning(f"{len(alerts):,} product-store location(s) need attention. Prioritize the largest order quantities first.")
        st.dataframe(_inventory_display(alerts), hide_index=True, use_container_width=True)

    st.subheader("Inventory recommendation table")
    st.caption("All quantities are units. Historical stockout rate is shown as a percentage of cleaned sales observations.")
    st.dataframe(_inventory_display(inventory_plan), hide_index=True, use_container_width=True)
    csv_bytes = inventory_plan.to_csv(index=False).encode("utf-8")
    st.download_button("Download inventory plan (CSV)", data=csv_bytes, file_name="inventory_plan.csv", mime="text/csv")

    with st.expander("Model explanation and assumptions"):
        left, right = st.columns(2)
        with left:
            st.markdown("**What the model uses**")
            st.markdown("- A time-aware Random Forest regressor\n- Prior-day and prior-week demand lags\n- Seven-day and 28-day rolling demand statistics\n- Calendar signals and product, category, and store identifiers")
            st.markdown("**How inventory is calculated**")
            st.markdown("Safety stock uses the selected 95% service-level z-score and recent demand variability. Reorder point equals lead-time demand plus safety stock; the target adds the review period.")
        with right:
            st.markdown("**Simulation assumptions**")
            st.markdown("- Synthetic daily demand for reproducible portfolio review\n- Weekly and yearly seasonality plus a gradual trend\n- Promotions create demand lifts\n- Optional stockout effects cap observed sales below latent demand\n- Current inventory defaults to recent seven-day demand × seven cover days")
            top_features = metrics.get("top_features", [])
            if top_features:
                st.markdown("**Most influential features in this run**")
                importance = pd.DataFrame(top_features)
                if {"feature", "importance"}.issubset(importance.columns):
                    importance["importance"] = importance["importance"].round(3)
                    st.dataframe(importance, hide_index=True, use_container_width=True)

    if "insights" in artifacts and not artifacts["insights"].empty:
        st.subheader("Business takeaways")
        st.dataframe(artifacts["insights"], hide_index=True, use_container_width=True)

    st.caption("Built with Python, pandas, scikit-learn, matplotlib, and Streamlit. This dashboard is a decision-support demo, not a live ordering system.")


if __name__ == "__main__":
    main()
