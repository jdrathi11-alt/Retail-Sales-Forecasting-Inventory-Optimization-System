import numpy as np
import pandas as pd

from src.data_simulation import simulate_sales_data
from src.forecasting import forecast_future, train_forecaster
from src.inventory import calculate_inventory_plan, inventory_status, normal_z_value
from src.preprocessing.features import clean_sales_data


def test_forecaster_uses_chronological_split_and_forecasts_all_keys(small_config):
    history = simulate_sales_data(small_config)
    model = train_forecaster(history, small_config)
    forecast = forecast_future(model, history, small_config.forecast_horizon)
    assert model.cutoff_date > pd.Timestamp(small_config.start_date)
    assert model.metrics["test_start"] == model.cutoff_date.strftime("%Y-%m-%d")
    assert len(forecast) == small_config.n_products * small_config.n_stores * small_config.forecast_horizon
    assert forecast["date"].min() > history["date"].max()
    assert (forecast["forecast_units"] >= 0).all()


def test_inventory_math_and_status():
    assert normal_z_value(0.95) > 1.6
    assert inventory_status(5, 10, 20) == "Reorder now"
    assert inventory_status(15, 10, 20) == "Monitor"
    assert inventory_status(25, 10, 20) == "Healthy"

    history = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10),
            "product_id": ["P1"] * 10,
            "category": ["C1"] * 10,
            "store_id": ["S1"] * 10,
            "units_sold": [10] * 10,
            "stockout": [0] * 10,
        }
    )
    forecast = pd.DataFrame(
        {
            "product_id": ["P1", "P1"],
            "category": ["C1", "C1"],
            "store_id": ["S1", "S1"],
            "forecast_units": [12.0, 12.0],
        }
    )
    plan = calculate_inventory_plan(forecast, clean_sales_data(history))
    row = plan.iloc[0]
    assert row["demand_during_lead_time"] == 12 * 7
    assert row["recommended_order_qty"] >= 0
    assert row["inventory_status"] in {"Reorder now", "Monitor", "Healthy"}
