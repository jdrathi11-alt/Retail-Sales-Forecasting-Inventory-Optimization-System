# Notebook convention

Use this folder for optional exploratory notebooks that support the project narrative, such as:

- `01_data_exploration.ipynb` for demand distributions, seasonality, promotion lift, and stockout inspection.
- `02_model_diagnostics.ipynb` for backtesting, residual review, and feature importance.
- `03_inventory_sensitivity.ipynb` for service-level, lead-time, and cover-day scenarios.

Keep reusable logic in `src/` and keep notebooks focused on analysis, visuals, and communication. Avoid committing large generated data or notebook outputs that can be recreated from `main.py`.
