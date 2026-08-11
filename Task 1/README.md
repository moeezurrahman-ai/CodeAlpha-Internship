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
