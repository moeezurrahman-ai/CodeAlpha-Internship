from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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

RANDOM_STATE = 42


# ==========================================================
# 1. Load Dataset
# ==========================================================

dataset = fetch_ucirepo(id=144)

X = dataset.data.features.copy()
y = dataset.data.targets.squeeze().copy()


# ==========================================================
# 2. Convert Target Labels
# ==========================================================

# UCI labels:
# 1 = Good credit
# 2 = Bad credit

y = pd.to_numeric(y, errors="coerce")
y = y.map({1: 1, 2: 0}).astype(int)


# ==========================================================
# 3. Feature Engineering
# ==========================================================

numeric_cols = X.select_dtypes(
    include=np.number
).columns.tolist()

if len(numeric_cols) >= 2:
    first_feature = numeric_cols[0]
    second_feature = numeric_cols[1]

    X["feature_interaction_1"] = (
        X[first_feature] * X[second_feature]
    )


# ==========================================================
# 4. Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================================
# 5. Identify Feature Types
# ==========================================================

categorical_cols = X_train.select_dtypes(
    exclude=np.number
).columns.tolist()

numeric_cols = X_train.select_dtypes(
    include=np.number
).columns.tolist()


# ==========================================================
# 6. Preprocessing
# ==========================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])


# ==========================================================
# 7. Define Models
# ==========================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# ==========================================================
# 8. Train Models
# ==========================================================

fitted_models = {}

for name, estimator in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", estimator)
    ])

    pipeline.fit(X_train, y_train)

    fitted_models[name] = pipeline

    print(f"{name} trained successfully.")


print("\nAll models trained successfully!")

# ==========================================================
# 9. Evaluate Models
# ==========================================================

results = []

for name, model in fitted_models.items():

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Bad Credit", "Good Credit"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")


# ==========================================================
# 10. Model Comparison
# ==========================================================

results_df = pd.DataFrame(results)

print("\n\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ==========================================================
# 11. Save Results
# ==========================================================

from pathlib import Path
import json
import joblib

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# Save model comparison
results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)


# Select best model based on ROC-AUC
best_name = results_df.iloc[
    results_df["ROC-AUC"].idxmax()
]["Model"]

best_model = fitted_models[best_name]


# Save best model
joblib.dump(
    {
        "model": best_model,
        "feature_columns": X.columns.tolist(),
        "target_mapping": {
            "0": "Bad credit risk",
            "1": "Good credit risk"
        }
    },
    OUTPUT_DIR / "credit_scoring_model.joblib"
)


# Save best model name
with open(
    OUTPUT_DIR / "best_model.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {"best_model": best_name},
        file,
        indent=2
    )


print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print("Model comparison: outputs/model_comparison.csv")
print("Best model:       outputs/credit_scoring_model.joblib")
print("Best model info:  outputs/best_model.json")

print(f"\nBest model: {best_name}")