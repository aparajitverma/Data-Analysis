import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN
from sklearn.neighbors import KNeighborsClassifier


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    try:
        enc = OneHotEncoder(sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(sparse=False)
    enc_data = enc.fit_transform(df[cat_cols])
    enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names_out(cat_cols), index=df.index)
    num_df = df.drop(columns=cat_cols)
    out = pd.concat([num_df, enc_df], axis=1)
    if 'Churn_Yes' in out.columns:
        if 'Churn_No' in out.columns:
            out.drop('Churn_No', axis=1, inplace=True)
        out.rename(columns={'Churn_Yes': 'Churn'}, inplace=True)
    return out


def pick_20_samples(X_test: pd.DataFrame, y_test: pd.Series, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    idx = np.array(X_test.index)
    if len(idx) < 20:
        sel_idx = idx
    else:
        sel_idx = rng.choice(idx, size=20, replace=False)
    return sel_idx


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    docs_dir = os.path.join(here, 'docs')
    os.makedirs(docs_dir, exist_ok=True)

    df = load_and_clean(csv_path)
    df_enc = encode(df)

    X = df_enc.drop('Churn', axis=1)
    y = df_enc['Churn']

    # Resample with SMOTEENN as in the best-performing experiment
    sm = SMOTEENN()
    X_res, y_res = sm.fit_resample(X, y)

    # Train/test split on resampled data
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Best model determined earlier: KNN with distance weighting
    model = KNeighborsClassifier(n_neighbors=3, weights='distance')
    model.fit(X_train, y_train)

    # Evaluate on full test set for reference
    y_test_pred_full = model.predict(X_test)
    test_accuracy_full = accuracy_score(y_test, y_test_pred_full)

    # Pick 20 different samples from the test set for display
    sample_idx = pick_20_samples(X_test, y_test, random_state=42)
    X_20 = X_test.loc[sample_idx]
    y_20 = y_test.loc[sample_idx]

    y_20_pred = model.predict(X_20)
    # Use predict_proba if available
    probs = None
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_20)[:, 1]

    acc_20 = accuracy_score(y_20, y_20_pred)

    # Save results
    out_path = os.path.join(docs_dir, 'BEST20_PREDICTIONS.txt')
    lines = []
    lines.append('Best Model 20-Sample Predictions (SMOTEENN KNN)')
    lines.append('==============================================')
    lines.append(f"Model: KNeighborsClassifier(n_neighbors=3, weights='distance')")
    lines.append(f"Full test accuracy: {test_accuracy_full:.3f}")
    lines.append('')
    lines.append(f"20-sample subset accuracy: {acc_20:.3f}")
    lines.append('')

    # Map back selected indices to raw-like explanations is non-trivial post-encoding; we will print encoded row index and true/pred labels.
    # y values are 0/1 where 1=churn
    for i, (idx, y_true, y_pred) in enumerate(zip(X_20.index, y_20.values, y_20_pred)):
        row_hdr = f"{i+1:02d}. RowIdx {idx} – Actual: {'Yes' if y_true==1 else 'No'}, Predicted: {'Yes' if y_pred==1 else 'No'}"
        if probs is not None:
            row_hdr += f" (Churn probability: {probs[i]:.2f})"
        lines.append(row_hdr)
    lines.append('')
    lines.append('Note: Indices refer to the resampled test set rows. For full contextual explanations, see docs/PREDICTIONS.txt in the main run.')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('Saved 20-sample predictions to:', out_path)
    print('Full test accuracy:', round(test_accuracy_full, 3))
    print('20-sample subset accuracy:', round(acc_20, 3))


if __name__ == '__main__':
    main()
