# Fraud-anomaly-detection in Credit Card Transactions
- Unsupervised anomaly detection project that flags fraudulent credit-card transactions without using any labels during training. The three detectors are Isolation Forecast, Local Outlier Factor and One-Class SVM- Assign a score to each transaction based on the perceived abnormality, which will only be used for the purpose of verifying against actual fraudulent transactions. 

## Problem Statement
- Since actual frauds are unusual (approximate 0.17% of transactions) and keep changing, honest detections of fraud are nearly difficult. This is a great example of  unsupervised learning process since only features are used by the models. The actual labels in “Class” (where 1 = fraud) are omitted and are only used for measuring the performance of the models.

- Because of the classes are imbalanced, the metric used is **PR-AUC** (Average Precision) rather than accuracy or ROC-AUC alone, alongside precision@k (k = number of true frauds)

- Evaluation is done in two stages: a naive baseline(where fitting and scoring on the same data) and a time-based train/test split , which trains on the earliest 70% of transactions and tests on the most recent 30% simulating how a model trained on historical data performs ongenuinely unseen, future transactions.

## EDA Findings:
- Fraud transactions exhibit a clear pattern of distribution of amount compared to authentic transactions, and a number of PCA features differentiate the two categories very effectively in terms of the mean difference and Pearson correlation with the class.
- All features are standardised( mean of 0 and std of 1), since the distance and margin based  detectors( LOF and One-Class SVM) are scale-sensitive.
- Time and Amount would dominate the PCA featuress due to their larger ranges

## Project Structure
- Run the notebook from top to bottom.
- It covers EDA, data cleaning and scaling, model evaluation and time-based split evaluation in one file.

## Dataset
- Source: ULB Credit Card Fraud Detection. Kaggle
- This project uses a sample: all 492 frauds kept + 20,000 genuine transactions sampled → 20,492 rows.
Features: 30, Time, Amount, and V1, V28 (anonymised PCA components). No raw fields are available for privacy reasons.

## Key Findings
- **One-Class SVM generalises best to unseen data**: overtaking Isolation Forest on PR-AUC under the time-split evaluation (0.0462 vs 0.0402), despite flagging far more transactions.
- **PR-AUC, not ROC-AUC, tells the real story**: under this level of class imbalance , all three detectors look broadly similar on ROC-AUC, but PR-AUC exposes meaningful gaps in actual usable precision, particularly for Local Outlier Factor.
- **precision@k / recall@k dropped to 0.0 for all models under the time split**: The models displayed no real fraud cases when applied to more recent transactions, but that does not imply they did not function as intended. The apparent "failure" to predict all fraud accurately was not due to a small sample size that resulted to 0. Rather, the most likely scenario for the mismatch of the predictions with the fraud cases comes from the fact that patterns of fraud change over time. Thus, the flags indicating the most suspicious behavior at a given time did not line up with what fraud actually looked like later on.  Nevertheless, it does not mean that the models did not work. In fact, you can see that the models still successfully distinguished between fraud and genuine transactions, although not at the extreme level of suspicion.

## Tech Stack

- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook Anomaly_Detection_Transaction.ipynb
```

Download `creditcard.csv` from the Kaggle and place it in the project folder before running.
