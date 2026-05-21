"""
Train the disease-risk regression model (ML Numeric block).

This script performs two *iterations* of training, mirroring the structure
required by the course:

* Iteration 1 — baseline.  Linear Regression vs. Random Forest on the raw
  numeric features (no disease class, no engineered features).
* Iteration 2 — improved.  Adds the predicted disease class as a one-hot
  feature, two engineered features (temp_humidity_index, n_balance), and
  compares Random Forest vs. Gradient Boosting with tuned hyper-parameters.

The best CV model is saved to `models/disease_risk_model.pkl` together with
a JSON file describing the feature ordering used by `app.py`.

Run:
    python train_numeric_model.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/plant_disease_risk_dataset.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
KFOLDS = 5


# ---------------------------------------------------------------------------
# Iteration 1 — baseline
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CATEGORICAL_FEATURES = ["disease"]


def iteration_1(df: pd.DataFrame) -> dict:
    print("\n=== Iteration 1: baseline (numeric features only) ===")
    X = df[NUMERIC_FEATURES].copy()
    y = df["risk_score"].copy()

    cv = KFold(n_splits=KFOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = "neg_root_mean_squared_error"

    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "RandomForest_200": Pipeline([
            ("model", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)),
        ]),
    }

    results = {}
    for name, pipe in models.items():
        rmse_cv = -cross_val_score(pipe, X, y, scoring=scoring, cv=cv, n_jobs=-1)
        r2_cv   = cross_val_score(pipe, X, y, scoring="r2", cv=cv, n_jobs=-1)
        results[name] = {
            "rmse_mean": float(rmse_cv.mean()),
            "rmse_std":  float(rmse_cv.std()),
            "r2_mean":   float(r2_cv.mean()),
        }
        print(f"  {name:<20s}  RMSE = {rmse_cv.mean():6.2f} ± {rmse_cv.std():.2f}   R² = {r2_cv.mean():.3f}")
    return results


# ---------------------------------------------------------------------------
# Iteration 2 — engineered features + disease one-hot
# ---------------------------------------------------------------------------

CROP_N_OPT = {"Tomato": 90, "Potato": 80, "Pepper": 70, "Apple": 50, "Corn": 100}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Temperature-humidity index (close to the standard greenhouse THI)
    df["temp_humidity_index"] = (
        df["temperature"] - (0.55 - 0.0055 * df["humidity"]) * (df["temperature"] - 14.5)
    )
    df["n_balance"] = df.apply(
        lambda r: r["N"] - CROP_N_OPT.get(r["crop"], 80), axis=1
    )
    return df


def iteration_2(df: pd.DataFrame) -> tuple[dict, Pipeline, list[str]]:
    print("\n=== Iteration 2: engineered features + disease one-hot ===")
    df = engineer(df)

    numeric = NUMERIC_FEATURES + ["temp_humidity_index", "n_balance"]
    feature_cols = numeric + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df["risk_score"].copy()

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    models = {
        "RandomForest_tuned": Pipeline([
            ("pre", pre),
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=14, min_samples_leaf=3,
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
        "HistGradientBoosting":   Pipeline([
            ("pre", pre),
            ("model", HistGradientBoostingRegressor(
                max_iter=400, max_depth=6, learning_rate=0.06,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    cv = KFold(n_splits=KFOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = "neg_root_mean_squared_error"

    results = {}
    fitted: dict[str, Pipeline] = {}
    for name, pipe in models.items():
        rmse_cv = -cross_val_score(pipe, X, y, scoring=scoring, cv=cv, n_jobs=-1)
        r2_cv   = cross_val_score(pipe, X, y, scoring="r2", cv=cv, n_jobs=-1)
        results[name] = {
            "rmse_mean": float(rmse_cv.mean()),
            "rmse_std":  float(rmse_cv.std()),
            "r2_mean":   float(r2_cv.mean()),
        }
        print(f"  {name:<20s}  RMSE = {rmse_cv.mean():6.2f} ± {rmse_cv.std():.2f}   R² = {r2_cv.mean():.3f}")
        pipe.fit(X, y)
        fitted[name] = pipe

    # Pick the lower-RMSE model as the production model.
    best_name = min(results, key=lambda k: results[k]["rmse_mean"])
    print(f"\nBest iteration-2 model: {best_name}")
    return results, fitted[best_name], feature_cols


# ---------------------------------------------------------------------------
# Error analysis on a hold-out test split
# ---------------------------------------------------------------------------

def hold_out_evaluation(df: pd.DataFrame, model: Pipeline, feature_cols: list[str]) -> dict:
    df = engineer(df)
    X = df[feature_cols].copy()
    y = df["risk_score"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=df["disease"]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2  = r2_score(y_test, pred)
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    print(f"\nHold-out 20% test:  RMSE = {rmse:.2f}   MAE = {mae:.2f}   R² = {r2:.3f}")

    # Per-disease error breakdown
    eval_df = pd.DataFrame({
        "disease": df.loc[X_test.index, "disease"].values,
        "y_true":  y_test.values,
        "y_pred":  pred,
    })
    eval_df["abs_err"] = (eval_df["y_pred"] - eval_df["y_true"]).abs()
    per_disease = eval_df.groupby("disease")["abs_err"].agg(["mean", "max"]).round(2)
    print("\nPer-disease MAE / max-abs-err on hold-out:")
    print(per_disease.to_string())
    return {"rmse": rmse, "mae": float(mae), "r2": float(r2),
            "per_disease": per_disease.reset_index().to_dict(orient="records")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {df['disease'].nunique()} disease classes.")
    print(f"Target stats: mean={df['risk_score'].mean():.2f}, std={df['risk_score'].std():.2f}")

    it1 = iteration_1(df)
    it2, best_model, feature_cols = iteration_2(df)
    holdout = hold_out_evaluation(df, best_model, feature_cols)

    # Re-fit the best model on the *entire* dataset for production.
    df_eng = engineer(df)
    best_model.fit(df_eng[feature_cols], df_eng["risk_score"])

    joblib.dump(best_model, MODEL_DIR / "disease_risk_model.pkl")
    with open(MODEL_DIR / "numeric_feature_columns.json", "w") as f:
        json.dump({
            "feature_columns": feature_cols,
            "numeric_columns": NUMERIC_FEATURES + ["temp_humidity_index", "n_balance"],
            "categorical_columns": CATEGORICAL_FEATURES,
            "disease_classes": sorted(df["disease"].unique().tolist()),
        }, f, indent=2)

    with open(MODEL_DIR / "numeric_training_report.json", "w") as f:
        json.dump({"iteration_1": it1, "iteration_2": it2, "hold_out": holdout},
                  f, indent=2)
    print("\nSaved model to models/disease_risk_model.pkl")
    print("Saved feature ordering to models/numeric_feature_columns.json")
    print("Saved training report to models/numeric_training_report.json")


if __name__ == "__main__":
    main()
