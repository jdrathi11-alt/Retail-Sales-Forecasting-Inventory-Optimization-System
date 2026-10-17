# Methodology and interview guide

This note is a compact guide for explaining the project in a portfolio review or interview. It mirrors the implementation without hiding the assumptions that make the demo reproducible.

## One-minute explanation

> I built a deterministic retail demand-planning workflow. It simulates daily sales with product, store, seasonal, promotion, and stockout signals; cleans the observations; creates lag, rolling, and calendar features; and trains a Random Forest with a chronological holdout. I then use the forecast and recent demand variability to calculate safety stock, reorder points, target stock, and recommended order quantities. A Streamlit dashboard presents the outputs in the order an operations user would need them: forecast, alerts, and action table.

## Why this modeling choice?

The Random Forest is a deliberate baseline rather than a claim that one algorithm solves retail forecasting. It handles nonlinear interactions between calendar variables and categorical product/store identifiers, is easy to fit locally, and exposes feature importance for a beginner-friendly explanation. The chronological holdout is more important than a random split here because future observations must not influence the evaluation of past training data.

Useful comparison points for an interview:

- A seasonal-naive baseline would establish whether the model beats a simple weekly pattern.
- Gradient boosting or probabilistic models could improve accuracy or uncertainty estimates.
- A production model would need backtesting across multiple rolling windows, not only one holdout.

## Feature design

The feature builder uses only information available before the target date:

- `lag_1` and `lag_7` capture recent and weekly persistence.
- `rolling_mean_7`, `rolling_std_7`, and `rolling_mean_28` summarize local level and volatility.
- Calendar signals capture weekly, monthly, annual, weekend, and trend effects.
- Product, category, and store identifiers are one-hot encoded and aligned between train and inference.

The rolling features shift the series before calculating the window. This prevents the current day’s target from leaking into its own feature.

## Inventory translation

For each product-store location:

```text
demand_during_lead_time = forecast_daily_demand × lead_time_days
safety_stock = z(service_level) × daily_demand_std × √lead_time_days
reorder_point = demand_during_lead_time + safety_stock
target_stock = forecast_daily_demand × (lead_time_days + review_period_days) + safety_stock
recommended_order_qty = max(target_stock − current_inventory, 0), rounded up
```

The default service level is 95%, which controls the safety-stock buffer. In the absence of a real inventory snapshot, the demo uses recent seven-day demand multiplied by seven cover days as a clearly labeled current-inventory estimate.

## Business interpretation

- **Reorder now:** current inventory is below the reorder point.
- **Monitor:** current inventory is above the reorder point but below target stock.
- **Healthy:** current inventory meets or exceeds target stock.

These labels are decision aids, not automated purchase orders. A real workflow would apply supplier, budget, shelf-life, and case-pack constraints before sending an order.

## Demo walkthrough

1. Run `streamlit run app/streamlit_app.py`.
2. If artifacts are missing, choose **Generate demo data** in the sidebar.
3. Start with the KPI cards: explain what each measure means and which one is an action metric.
4. Point out the dashed forecast-start marker on the demand chart.
5. Use the reorder-alert table to discuss prioritization by recommended order quantity.
6. Expand **Model explanation and assumptions** to discuss inputs, service level, lead time, and feature importance.
7. Download `inventory_plan.csv` to show how a dashboard recommendation can become an analyst handoff artifact.

## Questions to be ready for

### How did you avoid leakage?

The model split is chronological, and rolling statistics are built from shifted values so the current target cannot contribute to its own feature.

### Why forecast observed sales instead of true demand?

That is a known limitation of the educational baseline. Stockouts censor demand. A production extension would model lost sales or use inventory availability as an explicit signal.

### What would you monitor in production?

Forecast error by product and store, bias during stockouts and promotions, data freshness, feature drift, reorder recommendations, service-level attainment, and inventory turns.

### What is the most valuable next improvement?

I would first add a seasonal-naive benchmark and rolling-origin backtesting. That establishes whether added model complexity creates reliable business value before optimizing the model further.
