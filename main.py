"""Command-line entrypoint for the retail forecasting demo."""

from __future__ import annotations

import argparse

from src.config import PipelineConfig
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic retail sales forecasting and inventory optimization.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV, JSON, and PNG outputs")
    parser.add_argument("--days", type=int, default=365, help="Number of historical days to simulate")
    parser.add_argument("--forecast-horizon", type=int, default=14, help="Number of future days to forecast")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible data and model")
    parser.add_argument("--no-stockouts", action="store_true", help="Disable simulated stockout effects")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        random_seed=args.seed,
        n_days=args.days,
        forecast_horizon=args.forecast_horizon,
        include_stockout_effects=not args.no_stockouts,
        output_dir=args.output_dir,
    )
    result = run_pipeline(config)
    print(f"Pipeline complete: {len(result.clean_sales):,} cleaned sales rows")
    print(f"Test MAE: {float(result.model.metrics['mae']):.2f} units")
    print(f"Recommended order quantity: {int(result.inventory_plan['recommended_order_qty'].sum()):,} units")
    print(f"Outputs saved to: {config.output_path.resolve()}")


if __name__ == "__main__":
    main()
