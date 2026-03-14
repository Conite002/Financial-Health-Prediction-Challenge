# Financial Health Prediction Challenge

**Platform:** Zindi (DataOrg)

**Task:** Multi-class classification - predict the Financial Health Index (FHI) of small and medium enterprises (SMEs) across Southern Africa

**Primary metric:** Macro F1-score

**Secondary metric:** Log-loss

---

## Problem Statement

Access to formal financial services remains a critical barrier for SMEs in developing economies. This challenge uses survey data collected from SME owners in **Eswatini, Lesotho, Malawi, and Zimbabwe** to predict their Financial Health Index, defined as:

| Class | Description | Train prevalence |
|-------|-------------|-----------------|
| Low | Financially vulnerable, limited access, high stress | 65.3% |
| Medium | Moderate financial stability and access | 29.8% |
| High | Strong financial health, broad product access | 4.9% |

The severe class imbalance - particularly the 4.9% representation of the High class - is the central modeling challenge.

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `Train.csv` | 9,618 | Labeled SME survey responses |
| `Test.csv` | 2,405 | Unlabeled records for submission |
| `SampleSubmission.csv` | 2,405 | Required submission format |
| `VariableDefinitions.csv` | - | Feature descriptions |

**Features (38 raw)** span four domains:
- **Demographics:** owner age, sex, country
- **Business profile:** age, turnover, expenses, income
- **Financial product access:** mobile money, credit cards, insurance, loans, debit cards (responses: Have now / Used to have / Never had)
- **Attitudes and perceptions:** views on business environment, insurance, risk, growth

---

## Repository Structure

```
Financial Health Prediction Challenge/
├── data/
│   ├── Train.csv
│   ├── Test.csv
│   ├── SampleSubmission.csv
│   └── VariableDefinitions.csv
└── notebook/
    ├── Financial_Health_Prediction.ipynb   # Baseline pipeline (15 sections)
    ├── FHI_Optimization.ipynb              # v1 optimization (Optuna + two-stage)
    ├── FHI_Optimization_v2.ipynb           # v2 optimization (merged strategy)
    └── Starter Notebook.ipynb              # Competition starter
```

---

## Notebooks

### `Financial_Health_Prediction.ipynb` - Baseline Pipeline

End-to-end notebook covering:
1. Data loading and text normalization
2. Exploratory data analysis
3. Ordinal encoding with survey-aware category hierarchies
4. Feature engineering (ratios, composites, missing indicators)
5. XGBoost, LightGBM, CatBoost training with `StratifiedKFold`
6. Ensemble blending
7. SHAP feature importance per class
8. Submission generation

**OOF Macro F1: ~0.81**

---

### `FHI_Optimization_v2.ipynb` - Expert Optimization Pipeline

**Target: Macro F1 >= 0.91**

Builds on the baseline with a structured multi-technique strategy:

#### Feature Engineering
- **Financial ratios:** expense ratio, income-to-turnover, profit proxy, log-transforms
- **Product scoring:** Have now = 3, Used to have = 2, Never had = 1 (approach1 scale)
- **Composite scores:** `financial_access_score`, `insurance_product_score`, `banking_product_score`
- **Stress indicators:** `stress_score` (shutdown worry + cash flow + sourcing problems)
- **Credit vulnerability:** `credit_access_vulnerability = (1 - formal_access) * (0.5 + informal_access)`
- **Profitability margin:** `operational_profitability_margin` clipped to [-1, 1]
- **Heuristic label:** `fhi_rule_bin` - domain-knowledge rule approximating FHI from access + stress + profit thresholds
- **Country-contextual features:** per-country z-score and percentile rank of financial columns (fit on train, applied to test - no leakage)
- **Cross-interaction features:** `stress_x_expense_ratio`, `access_x_log_turnover`

#### Model Strategy

| Technique | Purpose |
|-----------|---------|
| XGBoost (Optuna, 100 trials) | Strong baseline, tuned hyperparameters |
| LightGBM (Optuna, 80 trials) | Fast iteration, handles categoricals well |
| HistGradientBoosting | NaN-native, diverse tree strategy |
| Two-Stage Cascaded Classifier | Stage 1: Low vs Non-Low; Stage 2: Medium vs High (High goes from 5% to 14% of stage-2 data) |
| SMOTE + KNN Imputation (inside CV) | Oversample minority classes without leakage; KNN imputer for correlated survey data |
| Isotonic Probability Calibration | Better-calibrated probabilities improve threshold optimization |
| OOF Threshold Optimization (Nelder-Mead) | Per-class threshold tuning: `argmax(proba / threshold)` |
| Stacking (Logistic Regression meta-learner) | Combines XGB + LGB + HGB OOF probability vectors |
| Country-Specific Model Clones | Separate LGB model per country; blended 60% global + 40% local |
| Ensemble Weight Optimization | Nelder-Mead optimization of blend weights on OOF Macro F1 |

#### Encoding
Ordinal encoding preserving survey response order:
- `HAVE_ORDER`: Never had < Used to have < Don't know < Have now
- `YES_NO_DK`: No < Don't know < Yes
- `OFFER_ORDER`: No < Yes, sometimes < Yes, always

---

## Environment

```
Python:     3.10 (conda env: ragml-py310)
Interpreter: /home/conite/anaconda3/envs/ragml-py310/bin/python

Core packages:
  xgboost        >= 1.7
  lightgbm       >= 4.0
  catboost        >= 1.2
  scikit-learn   >= 1.5
  imbalanced-learn >= 0.12
  optuna         >= 3.0
  shap           >= 0.44
```

---

## Key Design Decisions

**Class imbalance:** The High class (4.9%) requires structural intervention beyond `class_weight="balanced"`. The two-stage classifier and SMOTE work together: stage 2 sees ~14% High samples (vs 5% globally), and SMOTE is applied strictly inside CV folds to prevent leakage.

**Country heterogeneity:** Financial values across Eswatini, Lesotho, Malawi, and Zimbabwe are not directly comparable (different currencies, economic scales). Country z-score and percentile features normalize within-country standing without introducing cross-country scale bias.

**Ordinal encoding vs one-hot:** Survey responses follow natural hierarchies (e.g., financial product access). Encoding these as ordered integers rather than dummy variables preserves the information content and reduces dimensionality.

**`fhi_rule_bin`:** A domain-knowledge heuristic that approximates the FHI label from first principles. Treated as a model feature, not an intermediate artifact - the exclusion filter explicitly preserves it.

---

## Results

| Model | OOF Macro F1 | Notes |
|-------|-------------|-------|
| Baseline XGBoost | ~0.81 | No tuning, default thresholds |
| XGBoost (Optuna) | TBD after run | 100 Optuna trials |
| LightGBM (Optuna) | TBD after run | 80 Optuna trials |
| Two-Stage Classifier | TBD after run | Structural imbalance fix |
| Full v2 Ensemble | TBD after run | All techniques combined |

*Run `FHI_Optimization_v2.ipynb` end-to-end to populate results.*

---

## Submission

The final submission file is generated in `FHI_Optimization_v2.ipynb` cell 32 and saved to `notebook/submission_v2.csv`. Format:

```
ID,Target
ID_XXXXX,Low
ID_YYYYY,High
...
```
