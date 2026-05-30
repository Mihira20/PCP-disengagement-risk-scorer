# PCP-disengagement-risk-scorer
**Predicting Medicare Advantage member disengagement 
before the Annual Enrollment Period.**

## Problem Statement
A Medicare Advantage health plan covering 112,754 members 
needed to identify which members were at risk of disengaging 
from their Primary Care Physician before AEP. Without a 
data-driven approach, care coordinators were prioritizing 
outreach alphabetically, missing high-risk members entirely.

## Aproach
Built an entire ML pipeline on CMS synthetic public Use 
Files (SynPUF) that:
- Constructs a clean member cohort across 3 years of claims.
- Engineered 47 features across chronic disease burden,
  Utilization trends, demographics, and coverage patterns.
- Trained and evaluated multiple classification models.
- Explained predictions using SHAP for clinical stakeholders.
- Output is a prioritized outreach list tiered by risk level.


## Results
| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Dummy Baseline | 51.04% | 0.50 |
| Logistic Regression | 66.95% | 0.74 |
| Random Forest | 71.81% | 0.78 |

**Final model:** Random Forest with 50 estimators  
**Top risk drivers:** Carrier claim spend, 
outpatient utilization, chronic disease burden

## Risk Tiering
- High Risk: 13,977 members (top 10%)
- Medium Risk: 32,497 members (next 30%)
- Low Risk: 66,280 members (bottom 60%)

## Dataset
CMS Synthetic Public Use Files (SynPUF) — 
publicly available at Kaggle.
Download and place in `data/raw/` before running.

## How to Run
```bash
git clone https://github.com/Mihira20/PCP-disengagement-risk-scorer
cd PCP-disengagement-risk-scorer
pip install -r requirements.txt
jupyter notebook
```
Run notebooks in order: PCP_disengagement.ipynb -> Model_evaluation.ipynb -> shap.ipynb


## Tech Stack
Python, pandas, scikit-learn, SHAP, Streamlit, 
matplotlib, seaborn