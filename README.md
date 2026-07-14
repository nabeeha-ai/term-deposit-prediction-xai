# Term Deposit Subscription Prediction (Bank Marketing)

DevelopersHub Corporation — Data Science & Analytics Advanced Internship, Task 1.

## Objective
Predict whether a bank customer will subscribe to a term deposit as a result of a marketing campaign, and explain the model's predictions using SHAP.

## Dataset
Structured to match the **UCI Bank Marketing Dataset** schema (age, job, marital status, education, balance, housing/personal loan, contact type, campaign details, previous outcome, target `y`). A synthetic dataset with the same schema and realistic feature-target relationships is included at `data/bank_marketing.csv` (the notebook works identically on the real UCI CSV — just replace the file).

## Approach
1. Loaded and explored the dataset (shape, types, missing values, class balance).
2. Encoded all categorical features with `LabelEncoder`.
3. Trained two classifiers: **Logistic Regression** and **Random Forest** (class-weighted to handle imbalance).
4. Evaluated both models with **Confusion Matrix**, **F1-score**, and **ROC Curve / AUC**.
5. Used **SHAP** (`TreeExplainer`) to compute global feature importance and explain 5 individual predictions.

## Results & Findings
- Random Forest outperformed Logistic Regression on ROC-AUC and F1-score, capturing non-linear feature interactions.
- **Call duration**, **previous campaign success**, and **account balance** were the strongest predictors of subscription.
- SHAP explanations provide a transparent, auditable basis for why individual customers were flagged as likely subscribers.
- **Business recommendation:** prioritize clients with a successful previous campaign outcome and higher balances; use pre-call features (not call duration) to build the initial targeting list.

## Tech Stack
`pandas` · `numpy` · `scikit-learn` · `shap` · `matplotlib` · `seaborn`

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Repository Structure
```
task1-term-deposit-prediction/
├── data/
│   └── bank_marketing.csv
├── notebook.ipynb
├── requirements.txt
└── README.md
```
