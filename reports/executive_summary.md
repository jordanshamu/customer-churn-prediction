# Executive Summary: Customer Churn Prediction
## Early Warning System for Customer Retention

**Prepared by:** Jordan Shamukiga, Data Analyst  
**Date:** November 2026  
**Data:** IBM Telco Customer Churn, 7,043 customers (figures from `reports/churn_metrics.json`)  
**Audience:** Business Leadership, Customer Success, Marketing

---

## The Business Problem

**26.5% of customers** in the dataset churned. For the financial case we assume each churner we keep is worth **$500** (an assumption, not a measured figure).

Customers who churned share recognisable traits in the data: they are mostly on flexible month-to-month contracts, they have been customers for less than a year, they pay by electronic check rather than automatic payment, and they have not subscribed to any support or security add-on services.

---

## What We Built

We trained a machine learning model on 7,043 customer records (5,634 for training, 1,409 held out for testing) to score each customer's likelihood of churning, flag at-risk customers for intervention, and explain why each customer is at risk so the retention team can personalise the outreach.

What the model does not do: it cannot tell us whether a retention offer will actually work for any individual customer. It identifies who is *likely* to leave based on current patterns — the actual effectiveness of intervention still needs to be measured through a controlled experiment (see Recommended Next Steps).

We compared three modelling approaches and they performed about the same (ROC-AUC 0.844–0.847 in cross-validation). We chose XGBoost because its predicted probabilities were somewhat more accurate (lower Brier score), though all three overstate churn risk. On the held-out customers it has ROC-AUC 0.842, and at the chosen cut-off it catches 76.7% of churners, with 52.2% of flagged customers actually churning.

---

## What Drives Churn: Five Key Findings

**1. Contract type is the single strongest predictor.**  
Customers on month-to-month contracts churn at 42.7%. Customers on two-year contracts churn at 2.8%. The switching cost embedded in a longer contract is the most effective retention mechanism in the data.

**2. The first 12 months are the danger zone.**  
47.4% of customers with 12 months' tenure or less churned, against 14.0% of those past 24 months, and 55.5% of all churners left within their first year. Retention investment should be front-loaded toward new customers, not spread evenly across the base.

**3. Customers who pay by electronic check churn at 1.7× the average rate** (45.3%, vs. 16.0% for automatic payment).  
This is a correlation, not a proven cause. We don't know whether the payment method itself drives disengagement, or whether customers who choose e-check are already less committed for other reasons (perhaps they actively avoided auto-pay because they were keeping their options open). If the payment method is causal, the intervention is simple: make auto-pay enrollment frictionless. If it's just a proxy for a less committed customer profile, then switching their payment method won't help — you'd need to address the underlying disengagement. The model flags these customers either way, but the right response depends on which explanation is true.

**4. Service adoption is a powerful stickiness lever.**  
Among customers with at least one add-on service, churn falls from 45.8% with one service to 5.3% with all six. Customers with zero add-ons churn at 21.4%, because that group includes customers with no internet service, who churn at 7.4%. There's a confound: customers with many services also tend to be on longer contracts (97% of six-service customers are), so the effect isn't purely attributable to services alone.

**5. Higher monthly charges increase churn risk.**  
Churners paid $74 a month on average, against $61 for customers who stayed, and fiber optic customers churn at 41.9% vs. 19.0% for DSL. These customers are actively comparison-shopping and may respond well to loyalty pricing.

---

## The Retention Strategy: Tiered Interventions by Risk Segment

Not every at-risk customer warrants the same response. We segmented the customer base by contract type and tenure to create four risk tiers, each with a specific first action:

| Risk Tier | Who They Are | Actual churn rate | Day-1 Action |
|---|---|---|---|
| 🔴 Critical | Month-to-month, 0–6 months tenure | 55.2% (n = 1,413) | CS rep calls within 14 days of signup; satisfaction check, then offers 1-year contract at 15% discount + free TechSupport for 3 months |
| 🟠 High | Month-to-month, 7–12 months tenure | 42.0% (n = 581) | At month 9, send personalised email: $25 bill credit applied automatically + one-click link to upgrade to annual contract |
| 🟡 Medium | One-year contract, 0–12 months tenure | 10.5% (n = 124) | At month 6, email campaign: bundled add-on (streaming + security) at 20% intro rate for 6 months |
| 🟢 Low | Two-year contracts, all tenures | 2.8% (n = 1,695) | 60 days before contract expiry, send renewal offer with 5% loyalty discount locked for next term |

The ROI figures below cost every offer at the same $50 assumption.

---

## The Financial Case

**Without this model:** Retention efforts are untargeted or reactive. Customers leave before anyone reaches out, or offers are sent to customers who were never at risk — wasting budget.

**With this model:** Offers go to customers the model ranks as most likely to churn, at a cut-off chosen to balance offer cost against the value of customers kept. About half of flagged customers in the test set actually churned (52.2%), so some offers will still go to customers who would have stayed.

### Unit Economics

| Parameter | Value |
|---|---|
| Value kept per retained churner | $500 (assumption) |
| Cost of retention offer | $50, paid for every flagged customer (assumption) |
| Net value of a successful retention | $450 ($500 minus the offer) |
| Retention success rate | 35% (assumption; applies only to flagged customers who would have churned) |

Net value = 0.35 × $500 × (flagged customers who churn) − $50 × (all flagged customers).

### Sensitivity: The Retention Rate Assumption

The 35% retention success rate is an assumption, not a measured number from this company's campaigns. It drives the ROI, so it deserves scrutiny. On the 1,409 held-out customers, the model flags 550, of whom 287 churned. The offers cost $27,500 in total:

| Retention success | Retained value | Net value | ROI multiple (retained ÷ cost) |
|---|---|---|---|
| 15% | $21,525 | −$5,975 | 0.78× |
| **35% (assumption)** | **$50,225** | **$22,725** | **1.83×** |
| 50% | $71,750 | $44,250 | 2.61× |

At 15% the campaign loses money. Until the actual retention rate is measured through a controlled experiment, the 1.83× figure should be treated as an estimate, not a guarantee.

### Threshold Optimisation

The classification threshold was chosen to maximise net value on out-of-fold training predictions, not left at the statistical default. Under these assumptions, an offer to a customer who would have churned is worth 0.35 × $500 − $50 = $125 on average, and an offer to one who would have stayed costs $50.

The result: the threshold came out at 0.50, the same as the default, so in this run threshold tuning added no value over the default.

---

## How to Use the Model in Practice

**Monthly scoring:** Run new customer data through the model at the start of each billing cycle. The output is a ranked list of customers by churn probability.

**CRM integration:** Feed the risk scores into the CRM (Salesforce, HubSpot, etc.) so that the customer success team sees a churn risk flag on each account.

**Campaign trigger:** Any customer above the optimised threshold (documented in `reports/churn_metrics.json`) enters the appropriate retention workflow based on their segment.

**Monitoring:** Track actual churn rates among flagged vs. unflagged customers monthly to confirm model accuracy. Retrain quarterly as customer behaviour and competitive dynamics evolve.

---

## What This Project Does Not Cover

- **Real-time scoring:** The model is designed for batch monthly scoring. Real-time API deployment would require additional engineering work.
- **Causal inference:** The model identifies customers likely to churn given current conditions. It cannot tell us whether a specific retention offer will change the outcome for any individual.
- **External competitive factors:** Competitor pricing and promotions are not in the dataset.
- **Measured retention rates:** The ROI projections use assumed values, not this company's actual campaign performance data.

---

## Recommended Next Steps

1. **Immediate:** Implement the Critical and High-tier outreach programmes for the top-scoring current customers
2. **Month 1–2:** Integrate churn scores into CRM as a custom field; establish baseline churn rate for flagged vs. unflagged cohorts
3. **Month 2–3:** Run a controlled A/B test on the retention offer to measure true retention success rate — this is the single most important number for validating the ROI model
4. **Quarterly:** Retrain model with fresh data; update CLV and retention cost assumptions

---

*Full technical methodology, model evaluation results, and reproducible code available in the project notebook.*  
*Contact: Jordan Shamukiga | [datascienceportfol.io/jordanshamu](https://datascienceportfol.io/jordanshamu)*
