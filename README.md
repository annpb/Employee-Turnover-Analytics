# Employee Turnover Analytics

Predict employee turnover and uncover key attrition drivers using an end‑to‑end ML pipeline (EDA → clustering → classification) to enable proactive retention strategies.

## What It Does

- Loads and cleans the `HR_comma_sep.csv` dataset (target: `left`).  
- Encodes `salary` and department, imputes missing values, and builds a feature set covering satisfaction, evaluation, workload, accidents, promotions, and tenure.  
- Explores correlations, distributions, and turnover by number of projects.  
- Clusters employees who left (KMeans on satisfaction vs. evaluation) to reveal distinct leaver profiles.  
- Trains and compares Logistic Regression, Random Forest, and (optionally) XGBoost with 10‑fold stratified CV and SMOTE for class imbalance.  
- Selects the best model via a recall‑weighted heuristic, then evaluates on a held‑out test set (classification report, ROC‑AUC, confusion matrix, ROC curve).  
- Reports feature importance/coefficients to guide retention actions.

## Quick Start

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
# Optional:
pip install xgboost

# Run the notebook/script (ensure HR_comma_sep.csv path is correct)
```

The code uses `RANDOM_STATE = 42` for reproducibility.

## Outputs

- Correlation heatmap and feature distributions.  
- Turnover analysis by `number_project`.  
- KMeans clusters of leavers (satisfaction vs. evaluation).  
- Model metrics (accuracy, precision, recall, F1, ROC‑AUC).  
- Confusion matrix, ROC curve, and feature importance.

## How It Helps HR

- Scores employees by churn risk for targeted outreach.  
- Highlights high‑impact levers (e.g., workload, satisfaction, salary, promotions).  
- Supports data‑driven retention programs and A/B testing of interventions.

## Notes

- Demo on a single static dataset; for production, refresh data regularly and monitor drift.  
- Possible extensions: SHAP for interpretability, probability calibration, and a simple API/Streamlit app for scoring.
