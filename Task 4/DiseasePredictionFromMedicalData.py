"""
CodeAlpha Internship — Task 4: Disease Prediction from Medical Data
Dataset: UCI Breast Cancer Wisconsin (Diagnostic).

This project is educational and must not be used as a medical diagnostic tool.
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


RANDOM_STATE = 42

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================================
# 1. Load Dataset
# ==========================================================

def load_data():
    dataset = fetch_ucirepo(id=17)

    X = dataset.data.features.copy()
    y = dataset.data.targets.squeeze().copy()

    # UCI target:
    # B = Benign
    # M = Malignant
    y = y.astype(str).str.strip().map({
        "B": 0,
        "M": 1
    })

    if y.isna().any():
        raise ValueError(
            "Unexpected target labels found in the UCI dataset."
        )

    return X, y.astype(int)


# ==========================================================
# 2. Evaluate Model
# ==========================================================

def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }

    print(f"\n## {name}")
    print("-" * len(name))

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Benign",
                "Malignant"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print(
        f"Accuracy:  {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall:    {metrics['recall']:.4f}"
    )

    print(
        f"F1-Score:  {metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC:   {metrics['roc_auc']:.4f}"
    )

    return metrics


# ==========================================================
# 3. Main
# ==========================================================

def main():

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    X, y = load_data()

    print(
        f"Dataset shape: {X.shape}"
    )

    print("\nClass distribution:")

    print(
        y.value_counts().rename(
            index={
                0: "Benign",
                1: "Malignant"
            }
        )
    )


    # ------------------------------------------------------
    # Train-Test Split
    # ------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )


    # ------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------

    # The dataset contains numeric medical features.
    #
    # Median imputation handles possible missing values.
    # StandardScaler standardises feature values.
    #
    # Keeping preprocessing inside the pipeline prevents
    # information from the test set leaking into training.

    preprocessing = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])


    # ------------------------------------------------------
    # Define Models
    # ------------------------------------------------------

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Support Vector Machine": CalibratedClassifierCV(
            estimator=SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),
            method="sigmoid",
            cv=5
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            eval_metric="logloss"
        )
    }


    # ------------------------------------------------------
    # Train and Evaluate Models
    # ------------------------------------------------------

    results = []
    fitted = {}

    for name, estimator in models.items():

        pipeline = Pipeline([
            (
                "preprocessing",
                preprocessing
            ),
            (
                "model",
                estimator
            )
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        print(
            f"\n{name} trained successfully."
        )

        results.append(
            evaluate_model(
                name,
                pipeline,
                X_test,
                y_test
            )
        )

        fitted[name] = pipeline


    # ------------------------------------------------------
    # Model Comparison
    # ------------------------------------------------------

    results_df = pd.DataFrame(
        results
    ).sort_values(
        "roc_auc",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


    # ------------------------------------------------------
    # Save Model Comparison
    # ------------------------------------------------------

    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )


    # ------------------------------------------------------
    # Save Best Model
    # ------------------------------------------------------

    best_name = results_df.iloc[0]["model"]

    joblib.dump(
        {
            "model": fitted[best_name],
            "feature_columns": X.columns.tolist(),
            "target_mapping": {
                "0": "Benign",
                "1": "Malignant"
            }
        },
        OUTPUT_DIR / "disease_prediction_model.joblib"
    )


    # ------------------------------------------------------
    # Save Best Model Information
    # ------------------------------------------------------

    with open(
        OUTPUT_DIR / "best_model.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "best_model": best_name
            },
            file,
            indent=2
        )


    print(
        f"\nSaved model comparison: "
        f"{OUTPUT_DIR / 'model_comparison.csv'}"
    )

    print(
        f"Saved best model: "
        f"{OUTPUT_DIR / 'disease_prediction_model.joblib'}"
    )

    print(
        f"Best model info: "
        f"{OUTPUT_DIR / 'best_model.json'}"
    )

    print(
        f"\nBest model: {best_name}"
    )


# ==========================================================
# Program Entry Point
# ==========================================================

if __name__ == "__main__":
    main()