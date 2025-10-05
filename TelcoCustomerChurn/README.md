# Telco Customer Churn – Python Project

This project converts an exploratory notebook/script into a clean, runnable Python program for analyzing and modeling customer churn using the IBM Telco Customer Churn dataset.

The workflow includes:
- Data loading and cleaning
- Exploratory Data Analysis (EDA) with saved plots
- One-hot encoding of categorical features
- Baseline model training with simple hyperparameter search
- Imbalance handling using SMOTEENN and model re-evaluation
- Exported plots of model performance and confusion matrix


## Repository Structure
- `telco_churn.py` – Main executable script with the full workflow
- `requirements.txt` – Python dependencies
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` – Dataset (place alongside the script)
- `outputs/` – Auto-generated directory containing plots and results images
  - `missing_heatmap.png`
  - `countplot_*.png` (categorical distributions vs Churn)
  - `hist_tenure.png`, `hist_MonthlyCharges.png`, `hist_TotalCharges.png`
  - `kde_charges.png`
  - `model_scores.png` (baseline)
  - `model_scores.png` (SMOTEENN) – generated again during the imbalance experiment
  - `confusion_matrix.png` (from best SMOTEENN model)


## Quickstart
1) Create and activate a virtual environment (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3) Place the dataset in the same folder as the script (if not already):
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`

4) Run the pipeline:
```powershell
python .\telco_churn.py
```

5) Review results:
- Open the `outputs/` folder to view plots and metrics.


## What the script does
- Loads the Telco Churn CSV and coerces `TotalCharges` to numeric.
- Drops missing rows after coercion and removes `customerID`.
- Saves EDA plots:
  - Missing value heatmap
  - Count plots of categorical columns vs `Churn`
  - Histograms and KDE plots of key numeric features by `Churn`
- One-hot encodes categoricals, renames `Churn_Yes` to `Churn` (binary 0/1).
- Trains several classifiers (RF, GB, SVM, LR, KNN, DT, AdaBoost, XGBoost, NB) with light grids.
- Reports test accuracy and plots model comparison bars.
- Runs a second experiment using `SMOTEENN` to handle class imbalance and repeats model selection.
- Saves a confusion matrix for the best SMOTEENN model.


## Results (example from a recent run)
- Best baseline model: RandomForestClassifier (accuracy ≈ 0.795)
- Best SMOTEENN model: KNeighborsClassifier (accuracy ≈ 0.983)

Note: Exact numbers may vary across runs due to randomness and environment versions.


## Customization
- To show plots interactively as well as saving them, edit `main(show_plots=True)` in `telco_churn.py`.
- You can adjust model grids in `evaluate_models()` and `imbalance_experiment()`.
- To persist models, we can add joblib save/load hooks on request.


## Requirements
See `requirements.txt` for the exact list. Core packages:
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- imbalanced-learn
- xgboost


## License
Dataset is from IBM Sample Data Sets; please review IBM’s dataset terms. Project code is provided as-is for educational purposes.
