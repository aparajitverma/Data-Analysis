# Telco Customer Churn – From EDA to High-Accuracy Modeling

This project builds an end-to-end churn risk pipeline on the IBM Telco dataset. We clean and explore the data, engineer encodings, evaluate multiple models, address class imbalance with SMOTEENN, and select a high-performing classifier (KNN) that reaches ~0.98 test accuracy. Most importantly, we translate EDA signals and model probabilities into actionable retention playbooks that help businesses identify high‑risk customers early and intervene with targeted offers.

---

## Dataset and Goal
- IBM Telco Customer Churn dataset.
- Objective: Predict whether a customer will churn (Yes/No) and highlight the factors that influence churn, to inform retention strategy.

---

## Key EDA Takeaways (from generated images)
- **Tenure and Lifecycle**
  - Very low tenure (≤ 12 months) shows substantially higher churn in histograms and density plots.
  - Longer tenure segments display steadily lower churn; `TotalCharges` patterns largely mirror tenure.
- **Contract Structure**
  - Month-to-month has the highest churn bars in countplots by `Contract`.
  - One-year and Two-year contracts show visibly lower churn proportions.
- **Internet Service Type**
  - `Fiber optic` customers exhibit higher churn share than `DSL` in the countplots.
  - This pattern persists even when looking at distributions conditioned on churn.
- **Payment Method and Billing**
  - `Electronic check` stands out with higher churn bars relative to autopay methods (Credit card/Bank transfer) in countplots.
  - `PaperlessBilling = Yes` associates with higher churn in the plots; treat as a dataset-specific proxy rather than causal.
- **Add-on Support/Security**
  - `OnlineSecurity = No` and `TechSupport = No` align with higher churn bars; their presence reduces churn proportions.
- **Pricing Effects**
  - Higher `MonthlyCharges` correlate with higher churn in histograms and KDEs; mid-range charges look more stable.
  - `TotalCharges` differences largely reflect tenure; caution for collinearity.
- **Overlap and Ambiguity**
  - Distributions for churn vs non-churn overlap across several features (e.g., moderate `MonthlyCharges`, mid-tenure), indicating borderline segments where errors are expected even with strong models.

Additional EDA insights worth noting
- **Tenure bucketing**: Compare churn across 0–6, 7–12, 13–24, 25–60, 60+ months to highlight early-lifecycle spikes and late stability.
- **Contract × Billing interactions**: Highest-risk cluster is Month-to-month + PaperlessBilling + Electronic check; lowest-risk cluster is 1–2 year contract + non-paperless + autopay.
- **Payment method concentration**: Quantify the share of churners paying via Electronic check vs. autopay (Bank/Credit). Interpretable as payment friction/reliability.
- **Add-on bundles**: Core add-ons (OnlineSecurity, TechSupport) reduce churn more consistently than entertainment add-ons; bundle effects matter.
- **Price segmentation**: View churn by `MonthlyCharges` deciles; top bands carry more churn, especially under Month-to-month contracts.
- **TotalCharges vs Tenure**: `TotalCharges` largely mirrors tenure; treat as collinear when modeling/interpretation.
- **Demographic stabilizers**: `Partner` and `Dependents` may correlate with lower churn; treat as context features rather than primary levers.

References (see `outputs/`):
- `missing_heatmap.png`, `countplot_*.png`, `hist_tenure.png`, `hist_MonthlyCharges.png`, `hist_TotalCharges.png`, `kde_charges.png`.

---
## Images Explained (what to look for)
- `missing_heatmap.png`: Data completeness; a clean map means no missing values remain.
- `hist_*` and `kde_charges.png`: For `kde_charges.png`, the dual density plots show where churn Yes/No concentrate across `MonthlyCharges` and `TotalCharges`. Taller peaks indicate common ranges; overlapping shaded areas indicate ambiguous regions where both classes coexist. Use this to reason about price thresholds and where probability calibration/threshold tuning will be critical.
- `model_scores_baseline.png`, `model_scores_smoteenn.png`: Accuracy comparisons across models before/after imbalance handling.
- `confusion_matrix.png`: Error profile (TP/TN/FP/FN) for the best model.

## How the images informed modeling decisions
### Image Insights
- Missing heatmap confirmed no imputation needed after coercing `TotalCharges` and dropping NA.
- Countplots highlighted risk categories (Month-to-month, Fiber optic, Electronic check, No OnlineSecurity/TechSupport, PaperlessBilling=Yes) to preserve via one-hot encoding.
- Histograms/KDEs revealed overlap regions — we used probability-based evaluation and recommend threshold tuning.
- Baseline vs SMOTEENN score charts exposed imbalance effects — we applied SMOTEENN and re-evaluated models.
- Confusion matrix guided FP vs FN trade-offs for business thresholds.

## Modeling Approach (what we did)
### Modeling Steps
- One-hot encoding of categorical features; `Churn_Yes` → `Churn` (single binary target).
- Baseline train/test split followed by several classifiers with light hyperparameter search.
- Imbalance handling via **SMOTEENN** and fresh model evaluation on a new split.

### Artifacts to revisit
- Baseline chart: `outputs/model_scores_baseline.png`
- SMOTEENN chart: `outputs/model_scores_smoteenn.png`
- Confusion matrix (best SMOTEENN model): `outputs/confusion_matrix.png`

---

## Results Summary
- Baseline (no resampling)
  - Best model: RandomForestClassifier
{{ ... }}
- With SMOTEENN (class imbalance mitigation)
  - Best model: KNeighborsClassifier (`n_neighbors=3`, `weights='distance'`)
  - Test accuracy: ≈ 0.981–0.983 (varies slightly across runs)
- 20-sample probe (held-out test subset)
  - Subset accuracy: 0.950  
  - Details: `docs/BEST20_PREDICTIONS.txt`
## Interpretable Predictions (4-case snapshot) and Takeaways
- Source: `docs/PREDICTIONS.txt` (2 predicted Yes + 2 predicted No with plain-English reasons).
- Samples (abbrev):
  - Sample 1 – Actual Yes, Pred Yes (p=1.00)
  - Sample 2 – Actual Yes, Pred Yes (p=1.00)
  - Sample 3 – Actual No, Pred Yes (p=1.00) ← false positive in an overlap region
  - Sample 4 – Actual No, Pred No (p=0.00)

Key takeaways from the 4 samples
- High-risk signature: Low tenure + Month-to-month + (Fiber optic/Electronic check) + No OnlineSecurity/TechSupport + PaperlessBilling=Yes.
- Retention signature: Long tenure + 1–2 year contract + non-paperless + OnlineSecurity present.
- Overlap example (Sample 3): Multiple risk factors can appear in non-churners too; this justifies probability thresholds and business-calibrated cutoffs.

---

## Why Errors Still Occur
- Overlapping patterns: churners and non-churners share similar profiles in some ranges.
- Proxy variables: payment method and paperless billing shift probabilities but are not direct causes.
- Resampling/split variance and neighborhood sensitivity (KNN) introduce variability.
- Potential label noise and missing temporal context.

---

## Final Conclusion
- **Most influential signals**: tenure, contract type, internet service type, payment method, OnlineSecurity/TechSupport, paperless billing (proxy), and price levels (`MonthlyCharges`).
- **Best-performing setup**: After SMOTEENN, **KNN (n_neighbors=3, weights='distance')** delivers ~0.98 test accuracy and ~0.95 accuracy on a 20-sample probe, indicating strong generalization.
- **Operational guidance**: Calibrate decision thresholds to business costs, track precision/recall/F1/ROC-AUC, and consider probability calibration for risk-based actions.
- **Business impact**: Use image-driven insights to profile high-risk segments (early-tenure, month-to-month, fiber, electronic check, no support/security, high charges). The model’s probabilities let retention teams rank accounts, trigger contract incentives, autopay nudges, and security/support trials, and measure uplift.
- **ROI framing**: Estimate avoided churn revenue by converting risk scores to expected saves (probability of churn × customer value) and measuring campaign lift.
- **Next steps**: Enrich features (e.g., support tickets, usage trends, plan change recency), try ensembles and calibration, and implement monitoring for drift with periodic retraining.

---

 

## Credits
- Dataset: IBM Telco Customer Churn (public dataset).
- Libraries: pandas, scikit-learn, imbalanced-learn, seaborn, matplotlib, xgboost.
