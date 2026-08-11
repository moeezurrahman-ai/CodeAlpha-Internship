# CodeAlpha Machine Learning Internship

This repository contains my work completed during the **Machine Learning Internship at CodeAlpha**.

The internship focuses on practical machine learning development using Python and machine learning libraries, with projects covering data preprocessing, model development, supervised and unsupervised learning, model evaluation, and optimisation.

---

## Internship

- **Organisation:** CodeAlpha
- **Role:** Machine Learning Intern
- **Year:** 2026

---

## Internship Tasks

According to the internship task instructions, Machine Learning interns are required to complete **any 2 or 3 out of the 4 available tasks**.

The available tasks are:

| Task | Project | Status |
|---|---|---|
| Task 1 | Credit Scoring Model | ✅ Completed |
| Task 2 | Emotion Recognition from Speech | 🔲 Not Started |
| Task 3 | Handwritten Character Recognition | 🔲 Not Started |
| Task 4 | Disease Prediction from Medical Data | ✅ Completed |

---

# Completed Projects

## Task 1 — Credit Scoring Model

### Objective

Predict an individual's creditworthiness using historical financial data.

The project uses the **UCI Statlog (German Credit Data)** dataset and compares multiple classification algorithms.

### Models Implemented

- Logistic Regression
- Decision Tree
- Random Forest

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

### Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 67.50% | 83.78% | 66.43% | 74.10% | 0.7617 |
| Decision Tree | 62.50% | 81.55% | 60.00% | 69.14% | 0.6383 |
| **Random Forest** | **71.50%** | **83.74%** | **73.57%** | **78.33%** | **0.7769** |

**Best-performing model:** Random Forest

**Best ROC-AUC:** 0.7769

### Project Structure

```text
Task 1/
├── CreditScoringModel.py
├── requirements.txt
├── README.md
└── outputs/
    ├── model_comparison.csv
    ├── credit_scoring_model.joblib
    └── best_model.json

*Task 4 — Disease Prediction from Medical Data
*Objective

Build machine learning classification models to predict whether a breast tumour is benign or malignant using medical diagnostic features.

The project uses the UCI Breast Cancer Wisconsin (Diagnostic) Dataset and compares multiple classification algorithms.

Disclaimer: This project is for educational purposes only. It is not a medical diagnostic system and must not be used to make real-world medical decisions.

*Dataset

Dataset: Breast Cancer Wisconsin (Diagnostic)

Source: UCI Machine Learning Repository

569 samples
30 numerical features
357 benign cases
212 malignant cases

The dataset is fetched programmatically using the ucimlrepo library.

*Models Implemented
Logistic Regression
Support Vector Machine (SVM)
Random Forest
XGBoost

*Evaluation Metrics
Accuracy
Precision
Recall
F1-Score
ROC-AUC
Classification Report
Confusion Matrix

*Results
Model	                  Accuracy      Precision      Recall      F1-Score      ROC-AUC
Random Forest	            97.37%	     100.00%	     92.86%	      96.30%	      0.9980
Support Vector Machine	  98.25%	      97.62%	     97.62%	      97.62%	      0.9950
Logistic Regression	      97.37%	      97.56%	     95.24%	      96.39%	      0.9954
XGBoost	                  97.37%	     100.00%	     92.86%	      96.30%	      0.9934

Best-performing model according to ROC-AUC: Random Forest

Best ROC-AUC: 0.9980

Highest test accuracy and F1-score: Support Vector Machine

*Project Structure
Task 4/
├── DiseasePredictionFromMedicalData.py
├── requirements.txt
├── README.md
└── outputs/
    ├── model_comparison.csv
    ├── disease_prediction_model.joblib
    └── best_model.json

*Repository Structure
codealpha-internship/
│
├── Task 1/
│   ├── CreditScoringModel.py
│   ├── requirements.txt
│   ├── README.md
│   └── outputs/
│       ├── model_comparison.csv
│       ├── credit_scoring_model.joblib
│       └── best_model.json
│
├── Task 2/
│   └── ...
│
├── Task 3/
│   └── ...
│
├── Task 4/
│   ├── DiseasePredictionFromMedicalData.py
│   ├── requirements.txt
│   ├── README.md
│   └── outputs/
│       ├── model_comparison.csv
│       ├── disease_prediction_model.joblib
│       └── best_model.json
│
└── README.md

*Internship Requirements

The internship instructions require interns to:

Complete the assigned projects within the given timeframe.
Upload the complete source code to GitHub.
Share a project explanation video on LinkedIn with the GitHub repository link.
Submit completed tasks through the designated submission form.

A minimum of two or three tasks must be completed for internship completion according to the provided instructions.

*About

This repository serves as a record of my practical machine learning work during the CodeAlpha Machine Learning Internship.

Each project is developed as an independent task and includes its source code, documentation, dependencies, and relevant outputs where applicable.
