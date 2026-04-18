# Executive Summary: Customer Churn Prediction
## Early Warning System for Customer Retention

**Prepared by:** Jordan Shamukiga, Data Analyst  
**Date:** March 2026  
**Audience:** Business Leadership, Customer Success, Marketing

---

## The Business Problem

Roughly **1 in 4 customers** cancels their subscription before their contract expires. Each lost customer represents approximately **$600 in lifetime value**, and re-acquiring a lapsed customer costs 5 to 10 times more than keeping them in the first place.

Customers who eventually cancel show recognisable patterns weeks or months beforehand: they are on flexible month-to-month contracts, they have been customers for less than a year, they pay by electronic check rather than automatic payment, and they have not subscribed to any support or security add-on services.

---

## What We Built

We trained a machine learning model on 7,000 customer records to score each customer's likelihood of churning in the next billing cycle, flag at-risk customers for intervention, and explain why each customer is at risk so the retention team can personalise the outreach.

What the model does not do: it cannot tell us whether a retention offer will actually work for any individual customer. It identifies who is *likely* to leave based on current patterns — the actual effectiveness of intervention still needs to be measured through a controlled experiment (see Recommended Next Steps).

After comparing three modelling approaches, the best-performing model (XGBoost gradient boosting) correctly identifies churning customers with strong discriminative power across all tested thresholds.

---

## What Drives Churn: Five Key Findings

**1. Contract type is the single strongest predictor.**  
Customers on month-to-month contracts churn at approximately 43%. Customers on two-year contracts churn at just 3%. The switching cost embedded in a longer contract is the most effective retention mechanism in the data.

**2. The first 12 months are the danger zone.**  
Churn probability drops sharply after the first year. Customers who survive past 24 months are largely anchored. Retention investment should be front-loaded toward new customers, not spread evenly across the base.

**3. Customers who pay by electronic check churn at nearly twice the average rate.**  
This is a correlation, not a proven cause. We don't know whether the payment method itself drives disengagement, or whether customers who choose e-check are already less committed for other reasons (perhaps they actively avoided auto-pay because they were keeping their options open). If the payment method is causal, the intervention is simple: make auto-pay enrollment frictionless. If it's just a proxy for a less committed customer profile, then switching their payment method won't help — you'd need to address the underlying disengagement. The model flags these customers either way, but the right response depends on which explanation is true.

**4. Service adoption is a powerful stickiness lever.**  
Customers subscribed to zero add-on services churn at roughly 50%. Customers subscribed to four or more services churn at under 10%. Each additional service meaningfully reduces churn probability — though there's a confound: customers with many services also tend to be on longer contracts, so the effect isn't purely attributable to services alone.

**5. Higher monthly charges increase churn risk.**  
Customers paying above-average monthly bills — particularly on fiber optic internet — show elevated price sensitivity. These customers are actively comparison-shopping and may respond well to loyalty pricing.

---

## The Retention Strategy: Tiered Interventions by Risk Segment

Not every at-risk customer warrants the same response. We segmented the customer base by contract type and tenure to create four risk tiers, each with a specific first action:

| Risk Tier | Who They Are | Day-1 Action | Estimated Cost |
|---|---|---|---|
| 🔴 Critical | Month-to-month, <6 months tenure | CS rep calls within 14 days of signup; satisfaction check, then offers 1-year contract at 15% discount + free TechSupport for 3 months | $75–100 |
| 🟠 High | Month-to-month, 7–12 months tenure | At month 9, send personalised email: $25 bill credit applied automatically + one-click link to upgrade to annual contract | $50–75 |
| 🟡 Medium | One-year contract, <12 months tenure | At month 6, email campaign: bundled add-on (streaming + security) at 20% intro rate for 6 months | $25–40 |
| 🟢 Low | Two-year contracts, all tenures | 60 days before contract expiry, send renewal offer with 5% loyalty discount locked for next term | $15–20 |

---

## The Financial Case

**Without this model:** Retention efforts are untargeted or reactive. Customers leave before anyone reaches out, or offers are sent to customers who were never at risk — wasting budget.

**With this model:** Every retention dollar is directed toward customers the model has identified as genuinely at risk, calibrated by the cost of acting vs. the cost of losing them.

### Unit Economics

| Parameter | Value |
|---|---|
| Average customer lifetime value (CLV) | $600 |
| Cost of retention offer | $50 |
| Net value of successful retention | $450 (CLV saved minus offer) |
| Retention success rate | 35% (see sensitivity note below) |
| Cost of missing a churner | $500 (CLV lost) |

### Sensitivity: The Retention Rate Assumption

The 35% retention success rate is an industry benchmark, not a measured number from this company's campaigns. This assumption drives the entire ROI model, so it deserves scrutiny:

- At **15% retention success**: the ROI multiple drops to roughly **2×** — still positive, but much less attractive. The campaign is borderline for lower-risk segments.
- At **35% retention success** (baseline): the ROI multiple is **4–6×**.
- At **50% retention success**: the ROI multiple climbs to **7–8×**.

Until the actual retention rate is measured through a controlled experiment, the 4–6× figure should be treated as an estimate, not a guarantee.

### Threshold Optimisation

The model uses an **optimised classification threshold** rather than the statistical default of 0.5. Because the cost of missing a churner ($500) is 10× the cost of a wasted retention offer ($50), it's economically rational to lower the threshold and accept more false positives in exchange for catching more true churners.

The result: the optimised model generates materially higher net business value than the same model at its default threshold, from the same number of predictions.

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
- **Measured retention rates:** The ROI projections use industry benchmarks, not this company's actual campaign performance data.

---

## Recommended Next Steps

1. **Immediate:** Implement the Critical and High-tier outreach programmes for the top-scoring current customers
2. **Month 1–2:** Integrate churn scores into CRM as a custom field; establish baseline churn rate for flagged vs. unflagged cohorts
3. **Month 2–3:** Run a controlled A/B test on the retention offer to measure true retention success rate — this is the single most important number for validating the ROI model
4. **Quarterly:** Retrain model with fresh data; update CLV and retention cost assumptions

---

*Full technical methodology, model evaluation results, and reproducible code available in the project notebook.*  
*Contact: Jordan Shamukiga | [datascienceportfol.io/jordanshamu](https://datascienceportfol.io/jordanshamu)*
