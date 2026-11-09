# Customer Churn Prediction: Decision Tree, Random Forest and XGBoost

Predicting which telecom customers are likely to leave (churn), comparing a tuned Decision Tree, a Random Forest and XGBoost, and evaluating them with metrics that suit an imbalanced problem.

## Problem

A telecom company loses revenue when customers cancel. If we can flag customers likely to churn, the company can offer them a retention deal before they leave.

- **Task:** binary classification (`Churn`: Yes / No)
- **Key challenge:** the classes are imbalanced (about 27% of customers churn), so accuracy alone is misleading. A model that always predicts "no churn" would score about 73% accuracy while catching nobody.

## Dataset

- **Source:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size after cleaning:** 7,032 customers (5,625 train / 1,407 test)
- **Features:** demographics, account info (tenure, contract, payment method), services subscribed (internet, security, tech support), and monthly / total charges
- **Target:** `Churn`

## Notebooks

| Notebook | What it covers |
|---|---|
| `01_decision_tree_random_forest.ipynb` | Data cleaning, encoding, Decision Tree (baseline and tuned), depth comparison, cross-validation, Random Forest, threshold tuning, feature importance |
| `02_xgboost.ipynb` | XGBoost classifier with early stopping, imbalance handling, evaluation and comparison with the other models |

## Approach

1. **Cleaning:** dropped `customerID` (an identifier, not a behaviour), converted `TotalCharges` to numeric (blank values became missing and those rows were removed), and removed duplicates.
2. **Encoding:** one-hot encoded categorical columns (`drop_first=True`).
3. **Split:** 80% train / 20% test, stratified so the churn ratio is the same in both.
4. **Decision Tree:** baseline, then tuned with `max_depth`, `min_samples_split`, `min_samples_leaf` and `class_weight="balanced"`.
5. **Random Forest:** 200 trees with the same constraints.
6. **XGBoost:**
   - A validation set was carved out of the **training** data only, so the test set stayed unseen.
   - `scale_pos_weight` (non-churners / churners, about 2.8) handles the class imbalance.
   - Early stopping on validation ROC-AUC (50 rounds) chose the number of trees.
   - The model was then retrained on the full training set with that number of trees.
   - Main settings: `learning_rate=0.05`, `max_depth=3`, `subsample=0.8`, `colsample_bytree=0.8`.
7. **Evaluation:** accuracy, precision, recall, F1, confusion matrix and ROC-AUC on the same test set for every model.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree (tuned) | TODO | TODO | TODO | TODO | TODO |
| Random Forest | 0.763 | 0.535 | 0.816 | 0.646 | 0.863 |
| XGBoost | 0.763 | 0.535 | 0.826 | 0.649 | 0.866 |

> **TODO before publishing:** fill in the Decision Tree row by adding it to the comparison table in the notebook (`"Decision Tree": model` in the `models` dictionary), and take every number from one clean run (**Restart, then Run All**).

**XGBoost confusion matrix (test set, 1,407 customers):**

|  | Predicted stay | Predicted churn |
|---|---|---|
| **Actually stayed (1,033)** | 764 | 269 |
| **Actually churned (374)** | 65 | 309 |

The model catches 309 of the 374 real churners (recall 0.83). About 46% of its churn alerts are false alarms (precision 0.53). This trade-off comes from using `scale_pos_weight`, which pushes the model toward catching leavers.

## Key Finding

XGBoost and Random Forest perform almost the same here (ROC-AUC 0.866 vs 0.863, F1 0.649 vs 0.646). A gap this small on a 1,407-row test set is within noise, so I do not claim XGBoost is better. The result suggests the limit comes from the information in the data, not from the choice of model. More complex models are unlikely to improve it much without new features.

> **TODO:** add 5-fold cross-validated ROC-AUC for both models to confirm they are equivalent.

## Feature Importance (Random Forest)

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

- Tenure and contract type are the strongest signals. Newer customers on month-to-month contracts look like the highest-risk group.
- **Caveats:** importance shows what the model relies on, not the direction of the effect or any cause. `tenure`, `TotalCharges` and `MonthlyCharges` are correlated, so credit is shared between them.

> **TODO:** add XGBoost's `feature_importances_` and compare the top features with the Random Forest. Confirm effect directions with `df.groupby("Contract")["Churn"].mean()`.

## Business Trade-off

The default threshold of 0.5 is not always best. Lowering it catches more leavers but creates more false alarms, and raising it does the opposite. F1 treats both mistakes equally, but a business usually does not.

To judge whether the model is useful, compare it against two simple baselines: offering a retention deal to **everyone** and offering it to **nobody**. A model only adds value if it costs less than both under realistic numbers for the offer cost and the value of a retained customer. These costs are assumptions, not company data, so state them clearly if you use them.

## What I Learned

- Accuracy is misleading on imbalanced data. Precision, recall, F1 and ROC-AUC give a fuller picture.
- Random Forest reduces variance by averaging many independent trees, while boosting builds shallow trees one after another, each fixing the previous errors.
- Early stopping needs a separate validation set, so the test set stays truly unseen.
- A fancier model does not guarantee a better result. Here XGBoost and Random Forest tied.

## Limitations and Next Steps

- Add 5-fold cross-validation comparison and XGBoost feature importance (see TODOs above)
- Tune the decision threshold on the training data, and compare it against the baselines above
- Try SHAP values to see the direction of each feature's effect
- Engineer new features (tenure groups, number of services subscribed)
- Deploy the best model as an API with FastAPI and Docker

## How to Run

```bash
git clone <your-repo-url>
cd Churn_prediction
pip install -r requirements.txt
jupyter notebook
```

Open the notebooks and run all cells. The dataset file (`Customer_dataset.csv`) should be in the same folder; download it from the Kaggle link above if it is not included.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, Jupyter

## Project Structure

```
Churn_prediction/
├── Customer_dataset.csv
├── 01_decision_tree_random_forest.ipynb
├── 02_xgboost.ipynb
├── requirements.txt
└── README.md
```
