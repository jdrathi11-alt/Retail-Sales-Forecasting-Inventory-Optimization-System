# Portfolio Playbook

This document is the step-by-step student guide for building, running, publishing,
and explaining the Retail Sales Forecasting & Inventory Optimization System.

## 1. What the system solves

Retailers balance two expensive outcomes:

- **Stockout:** a customer wants a product but the store cannot fulfill the sale.
- **Overstock:** cash and shelf/warehouse space are tied up in slow-moving stock.

The project forecasts near-term daily demand at product-store level and converts
that forecast into inventory actions. The action vocabulary is intentionally
simple: **Reorder now**, **Monitor**, or **Healthy**.

The business workflow is:

```text
sales history -> data quality -> EDA -> leakage-safe features -> forecast
              -> lead-time demand -> safety stock -> reorder point
              -> recommended order quantity -> dashboard/report
```

A company would connect the same ideas to point-of-sale, inventory snapshots,
supplier lead times, purchase orders, promotions, prices, weather, holidays,
and warehouse transfers. This portfolio version uses synthetic data because it
has no access to a company's private systems.

## 2. Technology choices

### Option A: practical baseline (selected)

- Python, pandas, NumPy
- scikit-learn Random Forest
- matplotlib/seaborn
- pytest
- Streamlit for the optional dashboard
- No GPU
- Best for a first portfolio project: complete, reproducible, easy to explain

### Option B: statistical time-series version

- pandas, statsmodels, seasonal naive, exponential smoothing, SARIMA or Prophet
- No GPU
- Better for comparing classical time-series assumptions, but more tuning and
  model-specific reasoning

### Option C: advanced production-style version

- XGBoost/LightGBM, rolling-origin backtesting, MLflow, FastAPI, Docker,
  scheduled jobs, database storage, monitoring, and cloud deployment
- GPU normally not required for tree models
- Stronger production story but too much surface area for the first build

Option A is selected because it demonstrates the complete decision loop without
hiding the logic behind a complex platform.

## 3. Architecture and module responsibilities

```text
PipelineConfig
  -> src.data_simulation.simulate_sales_data
  -> src.preprocessing.features.clean_sales_data
  -> src.preprocessing.features.build_features
  -> src.forecasting.train_forecaster
  -> src.forecasting.forecast_future
  -> src.inventory.calculate_inventory_plan
  -> src.reporting.save_pipeline_outputs
  -> app/streamlit_app.py
```

| Layer | Output |
| --- | --- |
| Simulation | product-store-day rows with observed and latent demand |
| Cleaning | valid dates/keys, non-negative units, deduplicated grain |
| Features | lag-1, lag-7, shifted rolling averages/volatility, calendar flags |
| Forecasting | chronological holdout metrics and future demand |
| Inventory | safety stock, reorder point, target stock, order quantity, status |
| Reporting | CSV, JSON, PNG, plain-language insight rows |
| Dashboard | KPI cards, charts, alert table, complete recommendation table |

## 4. Phase plan and verification

| Phase | Work | Expected proof | Common mistake |
| --- | --- | --- | --- |
| 1. Setup | Create `.venv`, install requirements | imports succeed | IDE uses another Python |
| 2. Structure | Keep `src`, `tests`, `app`, `docs`, `images`, `outputs` separate | folder tree | mixing generated files with source |
| 3. Simulate | Run `python main.py` | `clean_sales.csv` | calling synthetic data real company data |
| 4. Clean | Parse dates, coerce numbers, aggregate duplicates | no negative sales/invalid keys | dropping rows without reporting assumptions |
| 5. EDA | Inspect trends, categories, stores, promotions, stockouts | saved charts | plotting without business interpretation |
| 6. Features | Add shifted lags and rolling demand | feature table | leakage from current target |
| 7. Forecast | Time-aware split, train, evaluate, forecast | `metrics.json`, `forecast.csv` | random split for time series |
| 8. Inventory | Apply lead time, service level, safety stock | `inventory_plan.csv` | confusing reorder point with order quantity |
| 9. Dashboard | Run Streamlit and review alerts | screenshot/video | showing only model score, not action |
| 10. Test | Run pytest and compileall | all checks pass | testing only the happy path |
| 11. Publish | README, screenshots, GitHub commits | public clone runs | committing secrets or `.venv` |
| 12. Present | Practice one-minute and technical explanations | interview notes | hiding limitations |

## 5. Installation and commands

### Windows PowerShell

```powershell
cd "Retail Sales Forecasting & Inventory Optimization System"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('Dependencies OK')"
python -m pytest -q
python main.py --output-dir outputs --days 365 --forecast-horizon 14 --seed 42
streamlit run app/streamlit_app.py
```

### macOS/Linux

```bash
cd "Retail Sales Forecasting & Inventory Optimization System"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('Dependencies OK')"
python -m pytest -q
python main.py --output-dir outputs --days 365 --forecast-horizon 14 --seed 42
streamlit run app/streamlit_app.py
```

The core CLI works without an external dataset. If you have a CSV later, keep
the required columns `date`, `product_id`, `category`, `store_id`, and
`units_sold`, then call the cleaning and feature functions before training.

## 6. Simulation assumptions

The simulator creates:

- Products and categories at a known, repeatable mapping.
- Stores with different demand multipliers.
- Product base demand and price differences.
- Weekly and annual seasonality.
- A gradual trend.
- Promotion flags with a demand lift.
- Poisson demand noise.
- Optional stockout censoring: observed sales are reduced below latent demand
  on a small number of days.

This lets a student explain demand variation without claiming access to private
retail data. The default model predicts observed sales, so stockouts are a
known source of bias. A production upgrade should model lost sales explicitly.

## 7. Inventory formulas

```text
z = standard-normal quantile(service_level)
demand_during_lead_time = forecast_daily_demand * lead_time_days
safety_stock = z * daily_std_units * sqrt(lead_time_days)
reorder_point = demand_during_lead_time + safety_stock
target_stock = forecast_daily_demand * (lead_time_days + review_period_days)
               + safety_stock
recommended_order_qty = ceil(max(target_stock - current_inventory, 0))
```

The system defaults to a 95% service level, 7-day lead time, 7-day review
period, and seven days of proxy inventory cover. If a real snapshot is available,
pass an `inventory_state` table containing the key columns plus
`on_hand_units` instead of using the proxy.

## 8. Proof-building strategy

Use one meaningful commit per analytical slice:

```text
chore: initialize project structure
feat: add synthetic retail data generator
feat: add data cleaning and leakage-safe features
feat: train time-aware demand forecasting model
feat: add inventory reorder point calculations
feat: add Streamlit inventory decision dashboard
docs: publish methodology and portfolio walkthrough
```

Save these assets in `images/`:

```text
01-dataset-preview.png
02-data-quality.png
03-sales-trend.png
04-category-sales.png
05-forecast-overview.png
06-actual-vs-predicted.png
07-inventory-plan.png
08-reorder-alerts.png
09-dashboard-overview.png
10-github-repository.png
11-readme-preview.png
```

For a 60–90 second screen recording: launch the dashboard, generate demo data,
show KPI cards, scroll to the forecast, open reorder alerts, expand assumptions,
and download the CSV recommendation table.

## 9. GitHub commands

Recommended repository:
`retail-sales-forecasting-inventory-optimization`

Recommended topics:
`python`, `pandas`, `scikit-learn`, `forecasting`, `retail-analytics`,
`inventory-optimization`, `supply-chain`, `streamlit`, `data-science`,
`machine-learning`

After creating an empty GitHub repository:

```bash
git init
git add .
git commit -m "feat: add retail forecasting and inventory optimization system"
git branch -M main
git remote add origin https://github.com/<username>/retail-sales-forecasting-inventory-optimization.git
git push -u origin main
```

Pin the repository after adding screenshots, a clean README, exact setup
commands, limitations, and a short demo. Do not commit virtual environments,
private data, API keys, or uncontrolled generated output.

## 10. Resume and interview notes

### Resume bullets

- Built a reproducible Python retail demand-planning pipeline that simulated
  product-store-day sales with trend, seasonality, promotions, and stockouts;
  engineered leakage-safe lag/rolling features and trained a time-aware Random
  Forest baseline.
- Translated forecasts into safety stock, reorder points, target stock, and
  non-negative replenishment quantities using service-level and lead-time
  assumptions.
- Delivered a Streamlit decision-support dashboard with forecast KPIs,
  inventory health, reorder alerts, downloadable recommendations, and
  automated CSV/JSON/PNG reporting; added pytest regression coverage.

### Questions to prepare for

1. **Why synthetic data?** No private retail system was available, so I made
   the assumptions explicit and deterministic rather than pretending public
   data represented a company's operations.
2. **Why a chronological split?** Random splits let future observations inform
   training and can overstate performance for time series.
3. **How did you prevent leakage?** Rolling statistics are calculated after a
   one-period shift, so the current target is not included in its features.
4. **Why Random Forest?** It handles nonlinear interactions and categorical
   identifiers while staying understandable and CPU-friendly.
5. **What does MAE mean?** Average absolute units of error on the holdout period;
   it is easier for a planner to interpret than only a percentage.
6. **Why can observed sales be biased?** A stockout can turn high demand into a
   low sales count. The simulator exposes this limitation.
7. **How is safety stock different from reorder point?** Safety stock is a
   variability buffer; reorder point is expected lead-time demand plus that
   buffer.
8. **What would you monitor in production?** Error/bias by product and store,
   stockout periods, data freshness, feature drift, service level, inventory
   turns, and recommendation adoption.
9. **What would you improve first?** Add a seasonal-naive benchmark and rolling
   backtesting before adding a more complex model.
10. **Would you automate ordering?** Not without supplier, budget, case-pack,
    shelf-life, approval, and exception rules.

## 11. Troubleshooting

- `ModuleNotFoundError`: activate `.venv`, then run
  `python -m pip install -r requirements.txt` using the same interpreter that
  runs `python main.py`.
- PowerShell activation error: use `Set-ExecutionPolicy -Scope Process Bypass`
  or activate with Command Prompt.
- Empty features: use at least 35 historical days; 28-day rolling features
  need sufficient history.
- Invalid date values: use ISO dates; the cleaning module coerces invalid
  values and removes unusable observations.
- Negative forecast: forecasts are clipped at zero; inspect input scale if
  values are implausibly high.
- Too many reorder alerts: the default on-hand value is a transparent proxy.
  Supply actual `on_hand_units` for a realistic scenario.
- Dashboard says outputs are missing: run `python main.py` or click
  **Generate demo data** in the sidebar.
- Git identity error: configure repository-local `git config user.name` and
  `git config user.email`.
- Push rejected: inspect `git remote -v` and reconcile an existing remote
  README; never force-push `main`.
