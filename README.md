# Customer Churn Prediction: Decision Tree vs Random Forest

Predicting which telecom customers are likely to leave (churn), comparing a tuned Decision Tree with a Random Forest, and evaluating them with metrics that suit an imbalanced problem.

## Problem

A telecom company loses revenue when customers cancel. If we can flag customers likely to churn, the company can offer them a retention deal before they leave.

- **Task:** binary classification (`Churn`: Yes / No)
- **Key challenge:** the classes are imbalanced (about 26% of customers churn), so accuracy alone is misleading. A model that always predicts "no churn" would score about 75% accuracy while catching nobody.

## Dataset

- **Source:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size after cleaning:** 7,032 customers
- **Features:** demographics, account info (tenure, contract, payment method), services subscribed (internet, security, tech support), and monthly / total charges
- **Target:** `Churn`

## Approach

1. **Cleaning:** dropped `customerID` (an identifier, not a behaviour), converted `TotalCharges` to numeric (blank values became missing and those rows were removed), and removed duplicates.
2. **Encoding:** one-hot encoded categorical columns (`drop_first=True`).
3. **Split:** 80% train / 20% test, stratified so the churn ratio is the same in both.
4. **Models:**
   - Decision Tree: baseline, then tuned with `max_depth`, `min_samples_split`, `min_samples_leaf` and `class_weight="balanced"`
   - Random Forest: 200 trees with the same constraints
5. **Tuning:** compared tree depths, used 5-fold cross-validation on the training data, and chose the decision threshold from cross-validated predictions (not from the test set).
6. **Evaluation:** accuracy, precision, recall, F1, confusion matrix and ROC-AUC.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree (depth 7, balanced) | 0.73 | 0.48 | 0.83 | 0.61 | n/a |
| Random Forest (threshold 0.50) | 0.75 | 0.50 | 0.82 | 0.62 | 0.843 |
| Random Forest (tuned threshold) | n/a | 0.58 | 0.74 | 0.65 | 0.843 |

> **TODO before publishing:** replace these numbers with the final values from your notebook, and fill in the `n/a` cells by running the evaluation code on each model.

**Overfitting check:** Random Forest train accuracy 0.779 vs test accuracy 0.746. The small gap shows the depth and leaf limits kept the model from memorizing the training data.

**Stability check:** 5-fold cross-validated F1 for the tuned tree ranged from 0.573 to 0.632 (mean 0.604), consistent with its test F1.

### Depth comparison (Decision Tree)

| Max depth | Precision | Recall | F1 |
|---|---|---|---|
| 3 | 0.41 | 0.91 | 0.56 |
| 5 | 0.48 | 0.77 | 0.59 |
| **7** | 0.48 | 0.83 | **0.61** |
| 10 | 0.47 | 0.77 | 0.58 |

Very shallow trees underfit (they flag almost everyone), and very deep trees start to overfit. Depth 7 gave the best balance.

## Key Findings: Feature Importance

Top features in the Random Forest:

| Feature | Importance |
|---|---|
| `tenure` | 0.190 |
| `Contract_Two year` | 0.134 |
| `TotalCharges` | 0.124 |
| `InternetService_Fiber optic` | 0.102 |
| `MonthlyCharges` | 0.069 |
| `PaymentMethod_Electronic check` | 0.057 |
| `Contract_One year` | 0.038 |
| `OnlineSecurity_Yes` | 0.035 |
| `TechSupport_Yes` | 0.030 |

- Tenure and contract type are the strongest signals. Newer customers on month-to-month contracts are the highest-risk group.
- Fiber optic customers, electronic-check payers, and customers without security or support add-ons also show up as higher risk.
- **Caveats:** feature importance shows what the model relies on, not the direction of the effect or any cause. `tenure`, `TotalCharges` and `MonthlyCharges` are correlated, so credit is shared between them.

> **TODO:** confirm the directions with `df.groupby("Contract")["Churn"].mean()` (and the same for `InternetService` and `PaymentMethod`), then add the actual churn rates here.

## Business Trade-off

The default threshold of 0.5 is not always best. Lowering it catches more leavers but creates more false alarms; raising it does the opposite.

As an illustration (assumed numbers, not real company data): if a retention offer costs ₹500 and a lost customer costs ₹10,000, a missed churner is 20 times more expensive than a wasted offer. In that case a lower threshold with higher recall can be cheaper overall even when its F1 is slightly lower. F1 treats both mistakes equally, so the threshold should be chosen from the real costs.

## What I Learned

- Accuracy is misleading on imbalanced data. Precision, recall, F1 and ROC-AUC give a fuller picture.
- Overfitting shows up as a gap between train and test scores, and limiting `max_depth` and leaf size reduces it.
- Hyperparameters and the threshold should be tuned with cross-validation on the training data, so the test set stays unseen until the end.
- Random Forest improves on a single tree mainly by reducing variance, though on this dataset the gain was modest.

## Limitations and Next Steps

- Add XGBoost / LightGBM and compare with the Random Forest
- Run `GridSearchCV` with `scoring="f1"` on the training data
- Engineer new features (tenure groups, number of services subscribed)
- Try permutation importance for a more reliable feature ranking
- Deploy the model as an API with FastAPI and Docker

## How to Run

```bash
git clone <your-repo-url>
cd Churn_prediction
pip install -r requirements.txt
jupyter notebook
```

Open the notebook and run all cells. The dataset file (`Customer_dataset.csv`) should be in the same folder; download it from the Kaggle link above if it isn't included.

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, Jupyter

## Project Structure

```
Churn_prediction/
├── Customer_dataset.csv
├── churn_decision_tree_random_forest.ipynb
├── requirements.txt
└── README.md
```
