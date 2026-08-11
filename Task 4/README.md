# Task 4 — Disease Prediction from Medical Data

## Overview

This project is part of my **CodeAlpha Machine Learning Internship**.

The objective of this task is to build machine learning classification models that predict whether a breast tumour is **benign** or **malignant** using medical diagnostic features.

The project uses the **UCI Breast Cancer Wisconsin (Diagnostic) Dataset** and compares multiple machine learning algorithms.

> **Disclaimer:** This project is for educational purposes only. It is not a medical diagnostic system and must not be used to make real-world medical decisions.

---

## Dataset

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository

The dataset contains:

- **569 samples**
- **30 numerical features**
- Two target classes:
  - Benign
  - Malignant

### Class Distribution

| Class | Samples |
|---|---:|
| Benign | 357 |
| Malignant | 212 |

The dataset is fetched programmatically using the `ucimlrepo` library.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- UCI Machine Learning Repository

---

## Machine Learning Models

Four classification algorithms were implemented and compared:

1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Random Forest**
4. **XGBoost**

---

## Machine Learning Pipeline

The project follows these steps:

1. Load the dataset from the UCI Machine Learning Repository.
2. Separate features and target labels.
3. Convert:
   - `B` → Benign
   - `M` → Malignant
4. Split the dataset into training and testing sets.
5. Apply median imputation.
6. Standardise numerical features.
7. Train multiple classification models.
8. Evaluate each model.
9. Compare model performance.
10. Save the best-performing model.

The dataset was divided into:

- **80% training data:** 455 samples
- **20% testing data:** 114 samples

---

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Classification Report
- Confusion Matrix

### Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 97.37% | 100.00% | 92.86% | 96.30% | **99.80%** |
| Support Vector Machine | **98.25%** | 97.62% | **97.62%** | **97.62%** | 99.50% |
| Logistic Regression | 97.37% | 97.56% | 95.24% | 96.39% | 99.54% |
| XGBoost | 97.37% | 100.00% | 92.86% | 96.30% | 99.34% |

Based on the **ROC-AUC score**, Random Forest was selected as the best-performing model.

---

## Project Structure

```text
Task 4/
│
├── DiseasePredictionFromMedicalData.py
├── README.md
├── requirements.txt
│
└── outputs/
    ├── model_comparison.csv
    ├── disease_prediction_model.joblib
    └── best_model.json

*Output Files
model_comparison.csv — performance comparison of all trained models.
disease_prediction_model.joblib — saved best-performing model.
best_model.json — information about the selected model.

*How to Run
1. Install dependencies
pip install -r requirements.txt
2. Run the Python script
python DiseasePredictionFromMedicalData.py

The dataset will be fetched automatically from the UCI Machine Learning Repository.

*Conclusion

The project successfully demonstrates a complete machine learning classification workflow for medical data, including data loading, preprocessing, model training, evaluation, comparison, and model persistence.

Among the evaluated models, Random Forest achieved the highest ROC-AUC score of 0.9980 and was therefore selected as the best model according to the project's evaluation criterion.

Note: The results are based on the UCI benchmark dataset and should not be interpreted as evidence of clinical performance.
