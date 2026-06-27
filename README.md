# 🚢 Titanic Survival Prediction using Random Forest

[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue)](https://www.kaggle.com/c/titanic)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Validated-orange)](https://scikit-learn.org/)

## 📌 Project Overview
This repository contains a production-ready Machine Learning pipeline developed to solve the classic **Titanic: Machine Learning from Disaster** competition on Kaggle. The goal is to predict passenger survival by leveraging socio-economic, demographic, and trip-related features. 

By employing a rigorous **Random Forest Classifier**, the pipeline achieves a stable test accuracy of **82.12%** while ensuring full experimental reproducibility.

---

## 🛠️ Key Technical Implementations

### 1. Robust Data Preprocessing & Imputation
* **Smart Imputation:** Handled missing continuous features (`Age`) utilizing the column **Median** to prevent leverage from outliers.
* **Structural Cleaning:** Safely dropped uninformative high-cardinality features (`Cabin`) exceeding a 70% missing-value threshold.
* **Categorical Encoding:** Leveraged Scikit-Learn's `LabelEncoder` to seamlessly transform textual categories (`Sex`, `Embarked`) into optimal mathematical scales.

### 2. Exploratory Data Analysis (EDA) & Insights
Comprehensive visualization using `Seaborn` and `Matplotlib` yielded crucial predictive insights:
* **Gender Bias:** Visual analysis confirmed a strong survival priority for female passengers (`Sex`), making it a critical branch split for tree models.
* **Socio-Economic Split:** Ticket class (`Pclass`) showcased an explicit survival trend favoring Class 1 over Class 3 passengers.

### 3. Model Architecture & Evaluation
* **Algorithm:** `RandomForestClassifier` initialized with 100 decision trees (`n_estimators=100`) and structured tree restrictions (`max_depth=5`) to heavily suppress overfitting.
* **Reproducibility:** Locked state seeding via `random_state=42` across data splitting (`train_test_split`) and model validation.

---

## 📊 Performance Metrics

The model was evaluated on an unseen 20% validation split (179 samples), yielding a solid global accuracy of **82.12%**.

### 🔹 Confusion Matrix
```text
[[92  13]   <-- [True Negatives (Died),  False Positives]
 [19  55]]  <-- [False Negatives,        True Positives (Survived)]
