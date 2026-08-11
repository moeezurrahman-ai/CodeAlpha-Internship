# CodeAlpha Internship — Task 1: Credit Scoring Model

## Overview

This project implements a machine learning-based credit scoring model using the **UCI Statlog (German Credit Data)** dataset.

The objective is to predict whether a credit applicant represents a **Good Credit** or **Bad Credit** risk based on their financial and personal attributes.

This is an educational machine learning project and is **not intended for real-world lending decisions**.

---

## Objective

The main objectives of this project are to:

- Load and explore a credit-risk dataset.
- Perform feature engineering.
- Preprocess numerical and categorical features.
- Train multiple classification algorithms.
- Evaluate the models using several classification metrics.
- Compare model performance.
- Save the best-performing trained model.

---

## Dataset

The project uses the **UCI Statlog (German Credit Data)** dataset, retrieved using the `ucimlrepo` library.

### Dataset characteristics

- **Samples:** 1,000
- **Original features:** 20
- **Target classes:** 2
- **Good credit:** 700 samples
- **Bad credit:** 300 samples
- **Missing values:** None

The original target labels were:

```text
1 = Good credit
2 = Bad credit

For this project, they were converted to:

1 = Good credit
0 = Bad credit

*Machine Learning Workflow

The project follows this workflow:

UCI German Credit Dataset
          ↓
Data Loading
          ↓
Target Encoding
          ↓
Feature Engineering
          ↓
Train-Test Split
          ↓
Data Preprocessing
          ↓
Model Training
          ↓
Model Evaluation
          ↓
Model Comparison
          ↓
Save Best Model

*Feature Engineering

A simple interaction feature was created by multiplying the first two numerical attributes:

feature_interaction_1 = Attribute2 × Attribute5

This adds an additional relationship between two numerical features for the classification models to use.

After feature engineering:

21 features

*Data Preprocessing

The dataset contains both numerical and categorical attributes.

*Numerical features

Numerical features were processed using:

Median imputation
StandardScaler

*Categorical features

Categorical features were processed using:

Most-frequent-value imputation
One-Hot Encoding

The preprocessing was implemented using a Scikit-learn ColumnTransformer and Pipeline.

After preprocessing and one-hot encoding, the feature representation contained:

62 features

*Train-Test Split

The dataset was divided using an 80/20 stratified split:

Dataset	    Samples
Training	   800
Testing	     200

Stratification was used to preserve the class distribution between the training and testing sets.

*Models

Three classification algorithms were trained and compared.

1. Logistic Regression

A linear classification model used as a baseline.

2. Decision Tree

A tree-based classifier capable of learning non-linear decision rules.

3. Random Forest

An ensemble of decision trees designed to improve predictive performance and generalisation.

*Model Evaluation

The models were evaluated using:

Accuracy
Precision
Recall
F1-Score
ROC-AUC
Classification Report
Confusion Matrix

*Results
Model	                  Accuracy      Precision      Recall      F1-Score      ROC-AUC
Logistic Regression	     67.50%	        83.78%	     66.43%	     74.10%	       0.7617
Decision Tree	           62.50%	        81.55%	     60.00%	     69.14%	       0.6383
Random Forest	           71.50%	        83.74%	     73.57%	     78.33%	       0.7769

*Best Model

Based on ROC-AUC, the Random Forest achieved the best overall performance in this experiment.

Random Forest results
Accuracy:   71.50%
Precision:  83.74%
Recall:     73.57%
F1-Score:   78.33%
ROC-AUC:    0.7769

Its confusion matrix on the 200-sample test set was:

[[40 20]
 [37 103]]

The model was therefore selected as the best-performing model and saved for future use.

*Project Outputs

The trained model and evaluation results are saved in the outputs directory:

outputs/
├── model_comparison.csv
├── credit_scoring_model.joblib
└── best_model.json
model_comparison.csv

Contains the evaluation metrics for all trained models.

credit_scoring_model.joblib

Contains the saved best-performing Random Forest pipeline, including preprocessing.

best_model.json

Stores the name of the selected best model.

*Technologies Used
Python
Pandas
NumPy
Scikit-learn
UCI ML Repository
Joblib

*Installation

Clone the repository and install the required dependencies:

pip install -r requirements.txt

*Running the Project

Run the Python script:

python CreditScoringModel.py

The program will:

Retrieve the dataset.
Perform preprocessing and feature engineering.
Train the three classification models.
Evaluate their performance.
Compare the models.
Save the model comparison and best model to the outputs directory.

*Project Structure
CA.InitialProject/
│
├── CreditScoringModel.py
├── README.md
├── requirements.txt
│
└── outputs/
    ├── model_comparison.csv
    ├── credit_scoring_model.joblib
    └── best_model.json

*Conclusion

This project demonstrates a complete supervised machine learning workflow for credit-risk classification.

Three classification algorithms were trained and evaluated. Among them, the Random Forest classifier achieved the strongest overall performance, obtaining an accuracy of 71.50% and a ROC-AUC of 0.7769 on the test set used in this experiment.

The trained model and evaluation results were saved as reusable project outputs.
