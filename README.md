# Shopper Intent Lab: Online Shopper Purchase Prediction

**Course:** Machine Learning, M.Tech (DSE), BITS Pilani WILP · **Assignment 2**
**Live app:** <PASTE STREAMLIT LINK>
**GitHub:** <PASTE GITHUB LINK>

## a. Problem statement

Predict whether an online-shop browsing session ends in a purchase (`Revenue` = 1) using page-visit, Google Analytics and visitor features. Five classification models (Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes and Support Vector Machine) are trained on the same data and compared on Accuracy, AUC, Precision, Recall, F1 and MCC. The models are then demonstrated in an interactive Streamlit app.

## b. Dataset description

| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository, *Online Shoppers Purchasing Intention Dataset* (https://archive.ics.uci.edu/dataset/468) |
| Instances | 12,330 sessions as downloaded; **12,205** after removing 125 exact duplicate rows |
| Features | 17 input features (11 numeric including the binary `Weekend`, 6 categorical) plus the target `Revenue` |
| Task | Binary classification: purchase (1) or no purchase (0) |
| Class balance | 84.37% no purchase (10,297) and 15.63% purchase (1,908): imbalanced, so accuracy alone is misleading |
| Missing values | None |

**Feature groups**
- *Page activity:* `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`
- *Google Analytics metrics:* `BounceRates`, `ExitRates`, `PageValues`
- *Calendar:* `SpecialDay`, `Month`, `Weekend`
- *Visitor / technical:* `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`

**Method**
- Stratified 80/20 split: 9,764 training rows and 2,441 test rows (382 purchases in the test set).
- Numeric columns are standardised; `Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region` and `TrafficType` are one-hot encoded. The preprocessing is saved inside each model pipeline, so the app accepts the raw CSV.
- Hyper-parameters were tuned with 3-fold stratified cross-validation on the training set only (scoring: ROC-AUC). The test set was used once, for the final evaluation.
- `class_weight="balanced"` is used for Logistic Regression, Decision Tree and SVM. kNN and Gaussian Naive Bayes have no such option.
- Exact duplicate rows were removed before splitting so the same session cannot appear in both train and test.
- The most correlated numeric feature with `Revenue` is `PageValues` (r = 0.49), followed by `ExitRates` (-0.20).

## c. GitHub repository

<PASTE GITHUB LINK>

```
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- ML_Assignment_2_Classification_final.ipynb
|-- model/            (saved pipelines: *.joblib and feature_info.json)
```

## d. Models used

### Comparison table (test set, 2,441 sessions)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8492 | 0.9108 | 0.5118 | 0.7932 | 0.6222 | 0.5534 |
| Decision Tree | 0.8378 | 0.9272 | 0.4896 | 0.8639 | 0.6250 | 0.5663 |
| kNN | 0.8759 | 0.8722 | 0.7762 | 0.2906 | 0.4229 | 0.4255 |
| Naive Bayes | 0.6518 | 0.7929 | 0.2888 | 0.8377 | 0.4295 | 0.3320 |
| Support Vector Machine | 0.8710 | 0.9080 | 0.5648 | 0.7644 | 0.6496 | 0.5826 |

Precision, Recall and F1 are for the positive class (purchase).

### Tuned hyper-parameters

| ML Model Name | Best hyper-parameters (3-fold CV, AUC) | CV AUC |
|---|---|---|
| Logistic Regression | C = 0.1 | 0.8988 |
| Decision Tree | max_depth = 5, min_samples_leaf = 50 | 0.9222 |
| kNN | n_neighbors = 31, weights = distance | 0.8467 |
| Naive Bayes | var_smoothing = 0.001 | 0.7824 |
| Support Vector Machine | C = 5, gamma = 0.01 (RBF kernel) | 0.9090 |

### Observations on model performance

Only about 84% of sessions are "no purchase", so a model that always predicts "no purchase" would already reach about 84% accuracy. Accuracy is therefore read together with AUC, Recall, F1 and MCC.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A steady all-rounder: Accuracy 0.8492, AUC 0.9108 (2nd best), Recall 0.7932 and MCC 0.5534, and 3rd on every other metric. Train and test scores are almost identical (AUC 0.9070 vs 0.9108), so it generalises without overfitting. Precision is only 0.5118, meaning about half of its "purchase" predictions are false alarms, which is the cost of using balanced class weights to catch more buyers. |
| Decision Tree | Best AUC (0.9272) and best Recall (0.8639), so it finds about 86% of the real buyers. The price is lower Precision (0.4896, 4th) and an Accuracy of 0.8378, which is actually below the roughly 84% obtained by always predicting "no purchase", so accuracy alone would understate this model. A shallow tree (max_depth = 5, min_samples_leaf = 50) kept overfitting small: train AUC 0.9348 vs test AUC 0.9272. |
| kNN | Highest Accuracy (0.8759) and Precision (0.7762) but the worst Recall (0.2906) and F1 (0.4229): it rarely predicts "purchase", so it is usually right when it does but misses about 71% of the buyers. Training Accuracy and AUC are 1.0000 against a test AUC of 0.8722 (k = 31, distance weighting), so it has memorised the training data. Its high accuracy mostly comes from the majority class (Recall for "no purchase" is 0.984). |
| Naive Bayes | The weakest model overall: lowest Accuracy (0.6518), AUC (0.7929), Precision (0.2888) and MCC (0.3320). Its Recall of 0.8377 is 2nd best, but only because it labels too many sessions as purchases (just 61.7% of non-buyers are classified correctly). Train and test scores are nearly identical (Accuracy 0.6533 vs 0.6518), so this is a poor fit rather than overfitting, plausibly because the independence assumption does not hold: the correlation heatmap shows strongly correlated features (BounceRates/ExitRates, ProductRelated/ProductRelated_Duration) and the one-hot columns are not Gaussian. |
| Support Vector Machine | Best F1 (0.6496) and MCC (0.5826), 2nd on Accuracy (0.8710) and Precision (0.5648), which gives the lowest average rank (2.17). It balances false alarms and missed buyers better than the other models (Recall 0.7644, 4th), and its AUC of 0.9080 is within 0.003 of Logistic Regression. The generalisation gap is small (train AUC 0.9289 vs test 0.9080), but it was the slowest to tune (39.2 s against 0.4 to 8.1 s for the others). |
| **Overall winner** | **Support Vector Machine.** It has the best MCC (0.5826) and F1 (0.6496), the two metrics that handle class imbalance best, and the lowest average rank (2.17). The margin over the Decision Tree is small (MCC 0.5663), and the Decision Tree is better on AUC (0.9272) and Recall (0.8639) and much faster to tune (1.7 s vs 39.2 s), so it would be the better choice if catching as many buyers as possible matters more than avoiding false alarms. |

## Streamlit app features

1. **Dataset upload:** upload a CSV of test data (or use the bundled `test_data.csv`); the app checks that all 17 feature columns are present.
2. **Model selection dropdown:** choose any of the five models.
3. **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model, with the majority-class accuracy shown as a reference point.
4. **Confusion matrix and classification report:** counts or share of each actual class, plus per-class precision, recall and F1.
5. Extras: ROC curve, an all-models comparison table on the uploaded data, and downloadable predictions.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Reproducing the results

Run `ML_Assignment_2_Classification_final.ipynb` from top to bottom (random seed 42). It downloads the dataset from UCI, trains and evaluates the five models, and writes `test_data.csv`, `requirements.txt` and the `model/` folder used by the app.
