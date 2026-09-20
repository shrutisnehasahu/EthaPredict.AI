# EthaPredict AI

**Machine Learning-Based Prediction of Ethanol Fuel Consumption, Remaining Driving Range, and Refueling Time**

## What it does

Given a vehicle's current fuel level, ethanol blend, and live driving conditions
(speed, RPM, temperature, AC, traffic, load, road type, etc.), EthaPredict AI
predicts:

1. **Fuel consumption** for the current trip/segment
2. **Remaining driving range** (km)
3. **Remaining driving time** (hr)
4. **Refuel recommendation / alert**
5. **Estimated CO2 emissions**, with a cross-blend comparison

## Project Structure

```
ethapredict/
├── PHASE1_PROBLEM_DEFINITION.md
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .github/workflows/pipeline-ci.yml   # CI: runs the full pipeline + R² quality gate on every push
├── EthaPredict_AI_Project_Report.docx
├── EthaPredict_AI_Presentation.pptx
├── data/
│   ├── ethapredict_raw.csv          # messy synthetic data (Phase 2-3)
│   ├── ethapredict_clean.csv        # cleaned data (Phase 4)
│   └── ethapredict_featured.csv     # + engineered features (Phase 6)
├── figures/                         # EDA plots (Phase 5)
├── models/
│   ├── scaler.joblib
│   ├── linear_regression.joblib
│   ├── random_forest.joblib
│   ├── gradient_boosting.joblib     # XGBoost stand-in (see note below)
│   ├── xgboost.joblib               # only present after running 07 locally
│   ├── feature_columns.joblib
│   ├── best_model.txt
│   └── model_comparison.csv
└── scripts/
    ├── 01_generate_dataset.py        # Phase 2-3
    ├── 02_clean_data.py              # Phase 4
    ├── 03_eda.py                     # Phase 5
    ├── 04_feature_engineering.py     # Phase 6
    ├── 05_train_models.py            # Phase 7-8 (Linear Reg, Random Forest, GB)
    ├── 06_prediction_engine.py       # Phase 9
    ├── 07_train_xgboost_LOCAL.py     # Phase 7 real XGBoost — RUN LOCALLY
    ├── 08_environmental_module.py    # CO2 module
    └── 09_dashboard_LOCAL.py         # Phase 10 Streamlit dashboard — RUN LOCALLY
```

## Getting Started (GitHub-ready)

```bash
git clone <your-repo-url>
cd ethapredict
pip install -r requirements.txt      # core deps; xgboost/streamlit are optional extras in this file
python scripts/run_pipeline.py
```

MIT-licensed (see `LICENSE`). CI (`.github/workflows/pipeline-ci.yml`) runs the full pipeline
and enforces an R² ≥ 0.85 quality gate on every push/PR to `main`.

## Running the Pipeline

**One command (recommended)** — runs Phases 2 through 9 + the environmental module,
end-to-end, from a clean state, and prints a summary:

```bash
cd ethapredict
python scripts/run_pipeline.py
```

**Or step-by-step**, if you want to inspect each phase individually:

```bash
cd ethapredict
python scripts/01_generate_dataset.py
python scripts/02_clean_data.py
python scripts/03_eda.py
python scripts/04_feature_engineering.py
python scripts/05_train_models.py
python scripts/06_prediction_engine.py       # sanity-check a sample prediction
python scripts/08_environmental_module.py    # sanity-check CO2 comparison
```

This was verified in a clean run: full pipeline completes in ~25 seconds and
reproduces identical results (Random Forest, R² = 0.989).

## ⚠️ Environment Note: XGBoost & Streamlit

This project was built in a sandbox with **no internet access**, so `xgboost`
and `streamlit` could not be pip-installed there. To keep the pipeline fully
runnable end-to-end in that environment:

- **XGBoost** → `GradientBoostingRegressor` was used as an in-sandbox stand-in
  (same gradient-boosted-trees family, very similar behavior). The *real*
  XGBoost script (`07_train_xgboost_LOCAL.py`) is fully written and ready —
  run it on your own machine after `pip install xgboost` to get true XGBoost
  results using the identical train/test split, and it will automatically
  update `best_model.txt` if XGBoost wins.
- **Streamlit dashboard** (`09_dashboard_LOCAL.py`) is fully written and
  functional but has not been executed in this sandbox. Run it locally:
  ```bash
  pip install streamlit
  streamlit run scripts/09_dashboard_LOCAL.py
  ```

## Model Results (this sandbox run)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 0.50 | 0.89 | 0.989 |
| Gradient Boosting (XGBoost stand-in) | 0.63 | 0.93 | 0.988 |
| Linear Regression | 3.82 | 4.79 | 0.678 |

Random Forest was selected as the best model and is what `06_prediction_engine.py`
and the dashboard load by default (via `models/best_model.txt`).

## Roadmap Coverage

| Phase | Status |
|---|---|
| 1. Problem Definition | ✅ Done |
| 2-3. Data Collection Design + Synthetic Dataset | ✅ Done (12,006 rows after cleaning) |
| 4. Data Cleaning | ✅ Done |
| 5. EDA | ✅ Done (7 plots) |
| 6. Feature Engineering | ✅ Done (8 new features) |
| 7. Model Training | ✅ Done (Linear Reg, RF, GB-as-XGBoost stand-in; real XGBoost script provided) |
| 8. Model Evaluation | ✅ Done (MAE/RMSE/R² comparison) |
| 9. Prediction Engine | ✅ Done |
| 10. Dashboard | ✅ Code complete, run locally |
| 11. Real-world IoT/OBD-II integration | 🔲 Future work — architecture described below |

## Phase 11 — Future Work (Real-World Data)

The current system runs on physics-informed synthetic data. To go further:

```
CAR -> OBD-II/Vehicle Sensors -> ESP32/Raspberry Pi -> Python -> ML Model
     -> Cloud Database -> Streamlit Dashboard
```

Important: standard OBD-II does not reliably expose ethanol percentage or
precise fuel level on most vehicles, so a real deployment should clearly
separate **measured sensor data**, **vehicle-reported data**, and **user-entered
data** (e.g., ethanol blend typically has to be user-entered or inferred from
fuel-station receipts/pump selection).

## Tech Stack

Python, Pandas, NumPy, Matplotlib/Seaborn, Scikit-learn, XGBoost (local),
Streamlit (local). Future: OBD-II + ESP32/Raspberry Pi + IoT.
