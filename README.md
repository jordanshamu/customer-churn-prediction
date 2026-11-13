# 📉 Customer Churn Prediction
### ML Pipeline with Business Cost-Benefit Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Lab-orange?logo=jupyter)](https://jupyter.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Selected%20Model-8e44ad)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Portfolio](https://img.shields.io/badge/Portfolio-datascienceportfol.io%2Fjordanshamu-2c3e50)](https://datascienceportfol.io/jordanshamu)

> Analyst: [Jordan Shamukiga](https://github.com/jordanshamu) · [datascienceportfol.io/jordanshamu](https://datascienceportfol.io/jordanshamu)

> **Data:** every result below comes from one run of the notebook on the IBM Telco Customer Churn file (7,043 rows, sha256 `16320c9c…3e55e91`), recorded in `reports/churn_metrics.json`.

---

## Business Context

This dataset comes from IBM's sample data library. The competitive dynamics in telecom have shifted since it was published (streaming unbundling, 5G, etc.), but the analytical patterns are surprisingly durable: contract structure, early-tenure vulnerability, and payment friction still drive churn in most subscription businesses today. I chose this dataset because it's a well-known benchmark with realistic class imbalance (26.5% churn), which means I can compare my results against published baselines honestly.

The core business problem hasn't changed either. A churn model that just predicts who leaves isn't useful on its own — what matters is whether you can intervene *before* they leave, at a cost that makes economic sense. That's why this project doesn't stop at model accuracy. The threshold optimisation and cost-benefit framing in §10 are where the actual business value lives.

---

## Key Results

### Model Performance

The model was chosen on 5-fold cross-validation on the training set (5,634 rows). The 20% stratified hold-out test set (1,409 rows) was scored only after that choice.

| Model | CV ROC-AUC | CV average precision | CV Brier | Test ROC-AUC | Test average precision | Test Brier |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.845 ± 0.013 | 0.658 ± 0.020 | 0.166 ± 0.005 | 0.842 | 0.636 | 0.169 |
| Random Forest | 0.847 ± 0.010 | 0.658 ± 0.017 | 0.164 ± 0.005 | 0.841 | 0.644 | 0.166 |
| **XGBoost (selected)** | **0.844 ± 0.009** | **0.659 ± 0.017** | **0.158 ± 0.004** | **0.842** | **0.654** | **0.160** |

> The three models are within noise of each other on ROC-AUC and average precision. XGBoost was selected because its CV Brier score was lower by more than 2 standard errors, though all three overpredict churn because of class weighting (see `10b_reliability_curve.png`).  
> Lower Brier is better. CV values are mean ± std across 5 folds. RF and XGBoost were tuned on the same folds, which slightly favours them.

At the cost-based threshold of **0.50** (chosen on out-of-fold training predictions; it happens to equal the default), XGBoost on the test set has precision 52.2%, recall 76.7% and F1 0.621 (287 true positives, 263 false positives, 87 false negatives).

### Business Impact

Test set (1,409 customers), threshold 0.50: 550 customers flagged, 287 of whom churned. Assumptions: $50 per offer, $500 kept per retained churner, and a retention success rate that applies only to flagged customers who churn.

| Retention success rate | 15% | 35% (assumption) | 50% |
|---|---|---|---|
| Offer cost | $27,500 | $27,500 | $27,500 |
| Retained value | $21,525 | $50,225 | $71,750 |
| Net value | −$5,975 | $22,725 | $44,250 |
| ROI multiple (retained value ÷ offer cost) | 0.78× | 1.83× | 2.61× |

- **Value lift from threshold optimisation vs. default 0.5:** $0, because the cost-based threshold came out at 0.50.
- **Full-dataset scoring (demonstration, includes training rows):** 2,743 of 7,043 customers flagged, 1,605 of whom churned; net value $143,725 at 35% (ROI multiple 2.05×). Details in `reports/churn_metrics.json`.

### Top Churn Drivers (SHAP rank for XGBoost, with churn rates from the data)

1. **Contract type: Month-to-month** (SHAP #1) — 42.7% churn vs. 11.3% for one-year and 2.8% for two-year contracts
2. **Tenure** (SHAP #2) — 47.4% churn for tenure ≤ 12 months vs. 14.0% after 24 months; 55.5% of churners left within their first 12 months
3. **Internet Service: Fiber Optic** (SHAP #3) — 41.9% churn vs. 19.0% for DSL
4. **Charges** (TotalCharges #4, avg_monthly_spend #5, MonthlyCharges #6) — churners pay $74/month on average vs. $61 for retained customers
5. **Payment: Electronic check** (SHAP #7) — 45.3% churn vs. 16.0% for automatic payment
6. **Gender: No effect** — 26.9% churn for women vs. 26.2% for men (chi-square p = 0.49). This is the most interesting null result in the dataset. A lot of telco marketing still segments by gender, and this data says it doesn't matter. Worth knowing what *not* to spend time on.

---

## Methodology

### 1. Exploratory Data Analysis
- Churn distribution and class imbalance assessment
- Categorical churn rates across contract, payment method, internet service, demographics
- Numeric feature distributions by churn status (tenure, charges)
- Service adoption heatmap — stickiness analysis (with confound acknowledgment)
- Correlation matrix

### 2. Feature Engineering

| Feature | Description |
|---|---|
| `tenure_band` | Categorical cohort grouping (0–6mo including tenure 0, 7–12mo, 13–24mo, 25–48mo, 49–72mo) |
| `monthly_charges_log` | Log-transform to reduce right skew for linear models |
| `avg_monthly_spend` | TotalCharges ÷ tenure — normalised spend per month |
| `num_services` | Count of add-on subscriptions — embeddedness proxy |
| `is_high_value` | Top-quartile by MonthlyCharges (cutoff computed on the training split only) |
| `is_month_to_month` | Binary flag — highest-risk contract type |
| `pays_by_echeck` | Electronic check — strong churn signal |

**Considered but dropped:** `contract_tenure_interaction` (contract type × tenure cross feature). The tree models find this interaction on their own through recursive splitting, and for logistic regression the one-hot contract dummies plus scaled tenure already capture the pattern adequately. Adding it just introduced redundancy without measurable lift in CV.

### 3. Preprocessing
- Missing value imputation (TotalCharges for 0-tenure new customers — filled with 0, not mean, because these are genuinely new accounts with no spending history)
- Binary encoding for Yes/No columns
- One-hot encoding for nominal categoricals
- Stratified 80/20 train-test split
- StandardScaler for Logistic Regression inside a Pipeline, so cross-validation refits it per fold (tree models receive raw data)

### 4. Multi-Model Comparison
- **Logistic Regression** — interpretable baseline with coefficient analysis
- **Random Forest** — ensemble baseline with MDI feature importance (note: MDI is biased toward continuous features on this dataset)
- **XGBoost** — gradient boosting with L1/L2 regularisation and scale_pos_weight
- GridSearchCV with 5-fold stratified CV for all ensemble models
- Model choice on 5-fold CV on the training set: average precision first, ties (paired fold difference < 2 standard errors) broken by Brier score, then by simplicity
- Test set used once after the choice: ROC-AUC, average precision, Brier, and a confusion matrix at the chosen threshold

### 5. Business Cost-Benefit Analysis
```
Net value at threshold t =
    0.35 × $500 × (flagged customers who churn)   ← retained value
  − $50 × (all flagged customers)                 ← offer cost
```
The $50 offer cost, $500 value per retained churner and 35% retention success rate are assumptions; the same figures are used in the threshold sweep and the ROI section. The sweep (0.05 to 0.95) runs on out-of-fold training predictions, and the test set is scored once at the chosen threshold.

### 6. Model Explainability (SHAP)
- Beeswarm summary plot — direction and magnitude of each feature across all customers
- Global bar chart — mean |SHAP| for feature ranking
- Individual waterfall — per-customer "why is this person high-risk?" explanation

### 7. Segment-Specific Risk & Retention Playbook
- Churn probability mapped onto Contract × Tenure segments
- Tiered retention interventions calibrated to risk level and economic value
- ROI on the test set at a 35% retention success rate (assumption — not measured), with 15% / 50% sensitivity
- Per-segment ROI from scoring all customers (demonstration only, since it includes training rows)

---

## Visualisations

All plots are auto-saved to `visualizations/` on notebook run.

| # | File | Description |
|---|---|---|
| 01 | `01_churn_distribution.png` | Churn vs. retained distribution |
| 02 | `02_churn_by_categorical.png` | Churn rate across 8 categorical features |
| 03 | `03_numeric_distributions.png` | Tenure, charges by churn status |
| 04 | `04_service_churn_heatmap.png` | Service adoption vs. churn rate |
| 04b | `04b_services_vs_churn.png` | Service count stickiness chart |
| 05 | `05_correlation_matrix.png` | Numeric feature correlations |
| 06a | `06a_cm_logreg.png` | Logistic Regression confusion matrix (out-of-fold training predictions, t = 0.5) |
| 06b | `06b_logreg_coefficients.png` | LR coefficient analysis |
| 07a | `07a_cm_rf.png` | Random Forest confusion matrix (out-of-fold training predictions, t = 0.5) |
| 08 | `08_feature_importance_*.png` | RF & XGBoost MDI feature importance |
| 08a | `08a_cm_xgb.png` | XGBoost confusion matrix (out-of-fold training predictions, t = 0.5) |
| 09 | `09_cost_benefit_threshold.png` | Net value vs. threshold (out-of-fold training predictions) |
| 09b | `09b_threshold_pr_value.png` | Precision/recall/value vs. threshold |
| 09c | `09c_cm_test_chosen.png` | Selected model confusion matrix on the test set at the chosen threshold |
| 10 | `10_model_comparison.png` | 5-fold CV ROC-AUC · average precision · Brier |
| 10b | `10b_reliability_curve.png` | Reliability curve (out-of-fold training predictions) |
| 11a | `11a_shap_beeswarm.png` | SHAP global beeswarm |
| 11b | `11b_shap_bar.png` | SHAP mean importance bar chart |
| 11c | `11c_shap_waterfall_highrisk.png` | SHAP waterfall for highest-risk customer |
| 12 | `12_segment_churn_heatmap.png` | Risk heatmap by contract × tenure segment |

---

## Project Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/                  ← WA_Fn-UseC_-Telco-Customer-Churn.csv or synthetic_telco.csv (not committed)
│   └── processed/            ← telco_churn_processed.csv (auto-generated)
├── notebooks/
│   └── churn_prediction.ipynb   ← Main analysis notebook (14 sections)
├── src/
│   ├── __init__.py
│   └── utils.py              ← Reusable ML pipeline utilities
├── reports/
│   ├── executive_summary.md  ← Non-technical stakeholder summary
│   └── churn_metrics.json    ← All model metrics (auto-generated)
├── visualizations/           ← All plots (auto-saved on notebook run)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Lab or Jupyter Notebook

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/jordanshamu/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Dataset

**Option A — Real dataset (used for the results above):**  
Source: IBM's [telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d) repository (same file as the [Kaggle copy](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)). 7,043 rows, 21 columns.

```bash
mkdir -p data/raw
curl -L -o data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

Expected sha256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`. The file is not committed (`data/raw/` is in `.gitignore`).

**Option B — Synthetic dataset (zero setup):**  
If the IBM file is missing, the notebook generates a synthetic dataset with the same schema and saves it as `data/raw/synthetic_telco.csv`. The data source (`ibm_telco` or `synthetic`), row count, file sha256 and random seed are written to `reports/churn_metrics.json`. Results on synthetic data are not comparable to the IBM results above.

### Run the Notebook

```bash
jupyter lab notebooks/churn_prediction.ipynb
```

Run all cells top-to-bottom. All visualisations save automatically to `visualizations/`, processed data to `data/processed/`, and metrics to `reports/churn_metrics.json`.

### Run Utils Directly

```python
from src.utils import load_and_validate, cost_benefit_analysis, evaluate_model

df = load_and_validate("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

---

## Technical Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Core language |
| pandas | 1.5+ | Data manipulation |
| numpy | 1.23+ | Numerical computing |
| scikit-learn | 1.1+ | ML pipeline, preprocessing, evaluation |
| xgboost | 1.7+ | Gradient boosting model |
| matplotlib | 3.6+ | Base visualisation |
| seaborn | 0.12+ | Statistical visualisation |
| shap | 0.41+ | Model explainability |
| jupyter | 3.0+ | Interactive notebook environment |

---

## Author

**Jordan Shamukiga**  
Data Analyst · Business Analyst · Product Analyst  

[![GitHub](https://img.shields.io/badge/GitHub-jordanshamu-181717?logo=github)](https://github.com/jordanshamu)
[![Portfolio](https://img.shields.io/badge/Portfolio-datascienceportfol.io-2c3e50)](https://datascienceportfol.io/jordanshamu)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
