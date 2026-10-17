import pandas as pd

from src.data_simulation import simulate_sales_data
from src.preprocessing.features import build_features, clean_sales_data, encode_features


def test_simulation_is_deterministic_and_has_expected_grain(small_config):
    first = simulate_sales_data(small_config)
    second = simulate_sales_data(small_config)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == small_config.n_days * small_config.n_products * small_config.n_stores
    assert {"promotion", "stockout", "true_demand"}.issubset(first.columns)
    assert (first["units_sold"] >= 0).all()


def test_cleaning_aggregates_duplicates_and_build_features_without_target_leakage():
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30),
            "product_id": ["P1"] * 30,
            "category": ["C1"] * 30,
            "store_id": ["S1"] * 30,
            "units_sold": list(range(30)),
        }
    )
    duplicate = raw.iloc[[0]].copy()
    duplicate["units_sold"] = 100
    cleaned = clean_sales_data(pd.concat([raw, duplicate], ignore_index=True))
    assert len(cleaned) == 30
    assert cleaned.loc[cleaned["date"] == pd.Timestamp("2024-01-01"), "units_sold"].item() == 100

    features = build_features(cleaned)
    first_complete = features.iloc[0]
    assert first_complete["lag_1"] == 27
    assert first_complete["lag_7"] == 21
    # Day 29's rolling mean is days 22-28, not day 29 itself.
    assert first_complete["rolling_mean_7"] == 24.0
    encoded = encode_features(features)
    assert encoded.shape[0] == features.shape[0]
    assert encoded.select_dtypes(exclude="number").empty
