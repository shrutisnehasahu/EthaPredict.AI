"""
run_pipeline.py — EthaPredict AI Master Pipeline
=====================================================
Runs the entire in-sandbox pipeline end-to-end, in order:

    Phase 2-3 -> Phase 4 -> Phase 5 -> Phase 6 -> Phase 7/8 -> Phase 9 -> Environmental

Usage:
    cd ethapredict
    python scripts/run_pipeline.py

This proves the whole system works together as one integrated project, not just
as isolated scripts. (Phase 10's Streamlit dashboard and the real XGBoost model
still need to run on your own machine — see README.md — but this script will
build and validate everything that feeds into them.)
"""

import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("Phase 2-3: Generate synthetic dataset", "01_generate_dataset.py"),
    ("Phase 4: Clean data", "02_clean_data.py"),
    ("Phase 5: Exploratory data analysis", "03_eda.py"),
    ("Phase 6: Feature engineering", "04_feature_engineering.py"),
    ("Phase 7-8: Train & evaluate models", "05_train_models.py"),
    ("Phase 9: Prediction engine sanity check", "06_prediction_engine.py"),
    ("Bonus: Environmental / CO2 module", "08_environmental_module.py"),
]


def run_step(label, script_name):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    script_path = os.path.join(SCRIPT_DIR, script_name)
    start = time.time()
    result = subprocess.run([sys.executable, script_path], cwd=SCRIPT_DIR,
                             capture_output=True, text=True)
    elapsed = time.time() - start
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print(f"\n❌ FAILED at: {label} (after {elapsed:.1f}s)")
        sys.exit(1)
    print(f"✅ Completed in {elapsed:.1f}s")


def main():
    overall_start = time.time()
    print("EthaPredict AI — Full Pipeline Run")
    for label, script_name in STEPS:
        run_step(label, script_name)

    total = time.time() - overall_start
    print(f"\n{'=' * 70}")
    print(f"🎉 FULL PIPELINE COMPLETE in {total:.1f}s")
    print("=" * 70)
    print("""
Artifacts produced:
  data/ethapredict_raw.csv        - messy synthetic data
  data/ethapredict_clean.csv      - cleaned data
  data/ethapredict_featured.csv   - feature-engineered data
  figures/*.png                   - 7 EDA plots
  models/*.joblib                 - trained models + scaler
  models/model_comparison.csv     - MAE/RMSE/R2 comparison
  models/best_model.txt           - which model the prediction engine uses

Next (on your own machine, where you can pip install):
  python scripts/07_train_xgboost_LOCAL.py   # real XGBoost, same split
  streamlit run scripts/09_dashboard_LOCAL.py # interactive dashboard
""")


if __name__ == "__main__":
    main()
