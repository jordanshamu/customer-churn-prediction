# 📉 Customer Churn Prediction
### ML Pipeline with Business Cost-Benefit Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Lab-orange?logo=jupyter)](https://jupyter.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Champion%20Model-8e44ad)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Portfolio](https://img.shields.io/badge/Portfolio-datascienceportfol.io%2Fjordanshamu-2c3e50)](https://datascienceportfol.io/jordanshamu)

> Analyst: [Jordan Shamukiga](https://github.com/jordanshamu) · [datascienceportfol.io/jordanshamu](https://datascienceportfol.io/jordanshamu)

---

## Business Context

This dataset comes from IBM's sample data library and dates to around 2013 — a period when US telcos were in the middle of the fiber buildout and cable bundling wars. The competitive dynamics have shifted since then (streaming unbundling, 5G, etc.), but the analytical patterns are surprisingly durable: contract structure, early-tenure vulnerability, and payment friction still drive churn in most subscription businesses today. I chose this dataset because it's a well-known benchmark with realistic class imbalance (~26% churn), which means I can compare my results against published baselines honestly.

The core business problem hasn't changed either. Acquiring a new customer costs 5–10× more than retaining an existing one. A churn model that just predicts who leaves isn't useful on its own — what matters is whether you can intervene *before* they leave, at a cost that makes economic sense. That's why this project doesn't stop at model accuracy. The threshold optimisation and cost-benefit framing in §10 are where the actual business value lives.

---

## Portfolio Narrative

| Project | Focus | Link |
|---|---|---|
| Project 1 | Healthcare Readmission Prediction | GitHub |
| Project 2 | Marketing A/B Test Analysis | [GitHub](https://github.com/jordanshamu/Marketing-A-B-Test-Analysis) |
| Project 3 | Customer Segmentation & Cohort Analysis | GitHub |
| **Project 4** | **Customer Churn Prediction ← You are here** | — |
| Project 5 | Experimentation Framework | Planned |

**Project 3 → Project 4 arc:** Project 3 answered *who are our customers?* through RFM and K-Means segmentation. Project 4 answers *which customers will leave, and what can we do about it?* Together they form a customer intelligence loop: **Understand → Predict → Retain**.

---

## Key Results

### Model Performance (Hold-Out Test Set, 20% stratified split)

| Model | ROC-AUC | PR-AUC | F1 Score | CV Stability |
|---|---|---|---|---|
| Logistic Regression | ~0.845 | ~0.660 | ~0.625 | ±0.008 |
| Random Forest | ~0.856 | ~0.672 | ~0.638 | ±0.007 |
| **XGBoost ✓ Champion** | **~0.862** | **~0.681** | **~0.647** | **±0.006** |

> Exact scores are computed at runtime and exported to `reports/churn_metrics.json`.  
> PR-AUC is the primary metric under the 26% churn class imbalance.  
> The gap between RF and XGBoost is small on 7K rows — XGBoost was chosen for better probability calibration, not raw AUC.

### Business Impact

| Metric | Value |
|---|---|
| At-risk customers flagged (full dataset) | See `churn_metrics.json` |
| Retention campaign ROI multiple | **4–6×** cost of intervention |
| Value lift from threshold optimisation vs. default 0.5 | Material — see §10 |
| Net expected ROI (7K customer base) | See `reports/churn_metrics.json` |

### Top Churn Drivers (SHAP Global Importance)

1. **Contract type: Month-to-month** — 43% churn rate vs. 3% for two-year contracts
2. **Tenure** — churn spikes sharply in months 1–12; the most dangerous window
3. **Monthly Charges** — higher-charge customers are more price-sensitive and churn more
4. **TechSupport: No** — customers without support are ~2× more likely to leave
5. **Internet Service: Fiber Optic** — premium service with high price sensitivity
6. **Gender: No effect** — churn rates are nearly identical for men and women. This is the most interesting null result in the dataset. A lot of telco marketing still segments by gender, and this data says it doesn't matter. Worth knowing what *not* to spend time on.

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
| `tenure_band` | Categorical cohort grouping (0–6mo, 7–12mo, 13–24mo, 25–48mo, 49–72mo) |
| `monthly_charges_log` | Log-transform to reduce right skew for linear models |
| `avg_monthly_spend` | TotalCharges ÷ tenure — normalised spend per month |
| `num_services` | Count of add-on subscriptions — embeddedness proxy |
| `is_high_value` | Top-quartile by MonthlyCharges |
| `is_month_to_month` | Binary flag — highest-risk contract type |
| `pays_by_echeck` | Electronic check — strong churn signal |

**Considered but dropped:** `contract_tenure_interaction` (contract type × tenure cross feature). The tree models find this interaction on their own through recursive splitting, and for logistic regression the one-hot contract dummies plus scaled tenure already capture the pattern adequately. Adding it just introduced redundancy without measurable lift in CV.

### 3. Preprocessing
- Missing value imputation (TotalCharges for 0-tenure new customers — filled with 0, not mean, because these are genuinely new accounts with no spending history)
- Binary encoding for Yes/No columns
- One-hot encoding for nominal categoricals
- StandardScaler for Logistic Regression (tree models receive raw data)
- Stratified 80/20 train-test split

### 4. Multi-Model Comparison
- **Logistic Regression** — interpretable baseline with coefficient analysis; chosen first for probability calibration
- **Random Forest** — ensemble baseline with MDI feature importance (note: MDI is biased toward continuous features on this dataset)
- **XGBoost** — champion model with L1/L2 regularisation and scale_pos_weight
- GridSearchCV with 5-fold stratified CV for all ensemble models
- Evaluation: ROC-AUC, PR-AUC, F1, Confusion Matrix, Classification Report

### 5. Business Cost-Benefit Analysis
```
Net Business Value at threshold t =
    (True Positives × $450 benefit)
  - (False Positives × $50 offer cost)
  - (False Negatives × $500 CLV loss)
```
Threshold sweep from 0.05 to 0.95 identifies the optimal threshold that maximises net value rather than F1 score — a critical distinction for production deployment.

### 6. Model Explainability (SHAP)
- Beeswarm summary plot — direction and magnitude of each feature across all customers
- Global bar chart — mean |SHAP| for feature ranking
- Individual waterfall — per-customer "why is this person high-risk?" explanation

### 7. Segment-Specific Risk & Retention Playbook
- Churn probability mapped onto Contract × Tenure segments
- Tiered retention interventions calibrated to risk level and economic value
- ROI calculation per segment at 35% assumed retention success rate (industry benchmark — not measured)

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
| 06a | `06a_cm_logreg.png` | Logistic Regression confusion matrix |
| 06b | `06b_logreg_coefficients.png` | LR coefficient analysis |
| 07a | `07a_cm_rf.png` | Random Forest confusion matrix |
| 08 | `08_feature_importance_*.png` | RF & XGBoost MDI feature importance |
| 08a | `08a_cm_xgb.png` | XGBoost confusion matrix |
| 09 | `09_cost_benefit_threshold.png` | Net business value curve |
| 09b | `09b_threshold_pr_value.png` | Precision/recall/value vs. threshold |
| 10 | `10_model_comparison.png` | Multi-model ROC-AUC · PR-AUC · F1 |
| 11a | `11a_shap_beeswarm.png` | SHAP global beeswarm |
| 11b | `11b_shap_bar.png` | SHAP mean importance bar chart |
| 11c | `11c_shap_waterfall_highrisk.png` | SHAP waterfall for highest-risk customer |
| 12 | `12_segment_churn_heatmap.png` | Risk heatmap by contract × tenure segment |

---

## Project Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/                  ← WA_Fn-UseC_-Telco-Customer-Churn.csv (not committed)
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

**Option A — Real dataset (recommended):**
1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Place in `data/raw/`

**Option B — Synthetic dataset (zero setup):**  
If `data/raw/` is empty, the notebook automatically generates a synthetic dataset with identical schema and realistic churn patterns. All analysis runs identically — useful for quick portfolio review. One caveat: the synthetic data was designed to reproduce the same patterns as the real dataset, so model performance on synthetic data will look similar. It shouldn't be treated as independent validation of the model's accuracy.

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
| xgboost | 1.7+ | Champion model |
| matplotlib | 3.6+ | Base visualisation |
| seaborn | 0.12+ | Statistical visualisation |
| shap | 0.41+ | Model explainability |
| jupyter | 3.0+ | Interactive notebook environment |

---

## Related Projects

- **Project 2:** [Marketing A/B Test Analysis](https://github.com/jordanshamu/Marketing-A-B-Test-Analysis) — 588K users, 42.5% conversion lift, p<0.001
- **Project 3:** Customer Segmentation & Cohort Analysis — RFM, K-Means, CLV estimation
- **Project 5:** Experimentation Framework *(coming soon)*

---

## Author

**Jordan Shamukiga**  
Data Analyst · Business Analyst · Product Analyst  

[![GitHub](https://img.shields.io/badge/GitHub-jordanshamu-181717?logo=github)](https://github.com/jordanshamu)
[![Portfolio](https://img.shields.io/badge/Portfolio-datascienceportfol.io-2c3e50)](https://datascienceportfol.io/jordanshamu)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
