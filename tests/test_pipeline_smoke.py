from pathlib import Path

from src.pipeline import run_pipeline


def test_end_to_end_pipeline_writes_expected_outputs(small_config):
    result = run_pipeline(small_config)
    expected = {
        "clean_sales.csv",
        "features.csv",
        "forecast.csv",
        "inventory_plan.csv",
        "business_insights.csv",
        "metrics.json",
        "forecast_overview.png",
        "inventory_status.png",
    }
    assert expected == set(path.name for path in result.output_paths.values())
    assert all(path.exists() and path.stat().st_size > 0 for path in result.output_paths.values())
    assert not result.forecast.empty
    assert not result.inventory_plan.empty
    assert Path(small_config.output_dir).is_dir()
