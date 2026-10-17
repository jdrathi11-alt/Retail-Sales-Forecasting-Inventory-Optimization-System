"""Synthetic retail sales data generation.

No external data is required. The generator deliberately exposes the signal
components (trend, weekly/yearly seasonality, promotions, and stockouts) so
that a beginner can relate model output back to business assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def simulate_sales_data(config: PipelineConfig | None = None) -> pd.DataFrame:
    """Generate deterministic daily sales for every product-store pair.

    ``units_sold`` is observed sales. When stockout effects are enabled, a
    stockout caps observed sales below latent ``true_demand``; this makes the
    optional effect visible without requiring an inventory simulator.
    """
    config = config or PipelineConfig()
    rng = np.random.default_rng(config.random_seed)
    dates = pd.date_range(config.start_date, periods=config.n_days, freq="D")

    categories = [f"Category_{i + 1}" for i in range(config.n_categories)]
    products = [f"Product_{i + 1:02d}" for i in range(config.n_products)]
    stores = [f"Store_{i + 1:02d}" for i in range(config.n_stores)]
    product_category = {product: categories[index % len(categories)] for index, product in enumerate(products)}
    product_base = dict(zip(products, rng.uniform(18, 75, size=len(products))))
    product_price = dict(zip(products, rng.uniform(8, 45, size=len(products))))
    store_multiplier = dict(zip(stores, rng.uniform(0.75, 1.30, size=len(stores))))
    category_multiplier = dict(zip(categories, rng.uniform(0.90, 1.15, size=len(categories))))

    rows: list[dict[str, object]] = []
    day_number = np.arange(config.n_days)
    day_of_week = dates.dayofweek.to_numpy()
    day_of_year = dates.dayofyear.to_numpy()
    weekly = 1 + 0.13 * np.sin(2 * np.pi * day_of_week / 7)
    yearly = 1 + 0.08 * np.sin(2 * np.pi * day_of_year / 365.25)
    trend = 1 + 0.30 * day_number / max(config.n_days - 1, 1)

    for product in products:
        category = product_category[product]
        for store in stores:
            base = product_base[product] * store_multiplier[store] * category_multiplier[category]
            promotion = rng.random(config.n_days) < config.promotion_probability
            promotion_multiplier = np.where(promotion, 1.35, 1.0)
            expected_demand = np.maximum(0.1, base * weekly * yearly * trend * promotion_multiplier)
            true_demand = rng.poisson(expected_demand).astype(int)
            stockout = np.zeros(config.n_days, dtype=int)
            units_sold = true_demand.copy()
            if config.include_stockout_effects:
                eligible = np.arange(config.n_days) >= 28
                stockout = (eligible & (rng.random(config.n_days) < config.stockout_probability)).astype(int)
                cap_ratio = rng.uniform(0.35, 0.75, size=config.n_days)
                units_sold[stockout == 1] = np.floor(
                    true_demand[stockout == 1] * cap_ratio[stockout == 1]
                ).astype(int)
            for index, date in enumerate(dates):
                rows.append({
                    "date": date,
                    "product_id": product,
                    "category": category,
                    "store_id": store,
                    "promotion": int(promotion[index]),
                    "stockout": int(stockout[index]),
                    "units_sold": int(units_sold[index]),
                    "true_demand": int(true_demand[index]),
                    "unit_price": round(float(product_price[product]), 2),
                })
    return pd.DataFrame(rows)
