"""
Prepare datasets for the Plant Disease Detector project.

This script does TWO things:

1. Generates `data/plant_disease_risk_dataset.csv`
   A merged numeric dataset that combines the structure of two real public
   datasets (the Kaggle "Crop Recommendation" dataset and weather/climate
   plant-disease risk data) with literature-grounded relationships between
   environmental conditions and plant-disease severity.

   We synthesise the merged dataset to keep training fully reproducible and
   to avoid Kaggle credentials in the build. The schema, feature ranges and
   plant-pathology relationships are taken from:
     * Atharva Ingle (Kaggle), "Crop Recommendation Dataset" — features
       N, P, K, temperature, humidity, ph, rainfall.
     * Agrios, "Plant Pathology", 5th ed. — temperature/humidity windows for
       fungal vs. bacterial diseases.

2. Documents how to download the PlantVillage image dataset for the
   computer-vision block (we don't ship the ~3 GB of images).

Run:
    python prepare_data.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SAMPLES_PER_DISEASE = 250


# ---------------------------------------------------------------------------
# 1. Disease catalogue
# ---------------------------------------------------------------------------
# 15 PlantVillage classes (5 crops, 1 healthy + diseased entries per crop).
# For each diseased class we encode the *optimal* conditions for disease
# progression (centre of the favourable window) and how strongly each driver
# matters. "healthy" classes have flat / neutral risk profiles.

DISEASE_PROFILES = {
    # crop, healthy_flag, optimal temp(°C), temp tolerance,
    # optimal humidity(%), humidity tolerance, ph optimum, ph tolerance,
    # rainfall optimum(mm), rainfall sensitivity, base_risk
    "Tomato___healthy":           dict(crop="Tomato",  healthy=True),
    "Tomato___Early_blight":      dict(crop="Tomato",  healthy=False, t_opt=27, t_tol=4, h_opt=90, h_tol=8,  ph_opt=6.3, ph_tol=0.8, rain_opt=120, rain_w=0.6, base=18),
    "Tomato___Late_blight":       dict(crop="Tomato",  healthy=False, t_opt=18, t_tol=4, h_opt=92, h_tol=6,  ph_opt=6.0, ph_tol=0.9, rain_opt=180, rain_w=0.9, base=22),
    "Tomato___Leaf_Mold":         dict(crop="Tomato",  healthy=False, t_opt=22, t_tol=4, h_opt=95, h_tol=4,  ph_opt=6.3, ph_tol=0.7, rain_opt=110, rain_w=0.5, base=15),
    "Tomato___Bacterial_spot":    dict(crop="Tomato",  healthy=False, t_opt=28, t_tol=4, h_opt=85, h_tol=8,  ph_opt=6.5, ph_tol=0.7, rain_opt=160, rain_w=0.8, base=20),
    "Potato___healthy":           dict(crop="Potato",  healthy=True),
    "Potato___Early_blight":      dict(crop="Potato",  healthy=False, t_opt=26, t_tol=5, h_opt=85, h_tol=8,  ph_opt=5.8, ph_tol=0.8, rain_opt=130, rain_w=0.6, base=17),
    "Potato___Late_blight":       dict(crop="Potato",  healthy=False, t_opt=17, t_tol=4, h_opt=93, h_tol=5,  ph_opt=5.8, ph_tol=0.8, rain_opt=200, rain_w=0.95, base=25),
    "Pepper___healthy":           dict(crop="Pepper",  healthy=True),
    "Pepper___Bacterial_spot":    dict(crop="Pepper",  healthy=False, t_opt=28, t_tol=4, h_opt=88, h_tol=7,  ph_opt=6.4, ph_tol=0.7, rain_opt=150, rain_w=0.85, base=19),
    "Apple___healthy":            dict(crop="Apple",   healthy=True),
    "Apple___Apple_scab":         dict(crop="Apple",   healthy=False, t_opt=18, t_tol=4, h_opt=92, h_tol=6,  ph_opt=6.0, ph_tol=0.8, rain_opt=170, rain_w=0.9, base=21),
    "Apple___Black_rot":          dict(crop="Apple",   healthy=False, t_opt=24, t_tol=5, h_opt=88, h_tol=8,  ph_opt=6.2, ph_tol=0.8, rain_opt=140, rain_w=0.7, base=18),
    "Corn___healthy":             dict(crop="Corn",    healthy=True),
    "Corn___Common_rust":         dict(crop="Corn",    healthy=False, t_opt=22, t_tol=5, h_opt=88, h_tol=7,  ph_opt=6.2, ph_tol=0.9, rain_opt=130, rain_w=0.6, base=16),
}


# ---------------------------------------------------------------------------
# 2. Synthetic feature ranges (matching the Kaggle Crop Recommendation dataset)
# ---------------------------------------------------------------------------

FEATURE_RANGES = {
    # nitrogen (mg/kg), phosphorus, potassium
    "N":           (0, 140),
    "P":           (5, 145),
    "K":           (5, 205),
    # weather
    "temperature": (8, 44),     # °C
    "humidity":    (15, 100),   # %
    "ph":          (3.5, 9.5),
    "rainfall":    (20, 300),   # mm / month
}

CROP_N_OPTIMUMS = {  # rough N optima per crop (mg/kg)
    "Tomato": 90,
    "Potato": 80,
    "Pepper": 70,
    "Apple":  50,
    "Corn":   100,
}


def _gauss_weight(x: float, mu: float, sigma: float) -> float:
    """0..1 weight, 1 at the optimum."""
    return float(np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)))


def compute_risk(profile: dict, row: dict) -> float:
    """Disease-progression risk in 0..100 based on environmental drivers."""
    if profile.get("healthy"):
        # Healthy class: low background risk, mildly elevated by extreme stress.
        stress = abs(row["temperature"] - 22) / 25 + abs(row["humidity"] - 70) / 100
        return float(np.clip(8 + 6 * stress + np.random.normal(0, 2), 0, 20))

    # Diseased class — weighted product of Gaussian fits to each driver.
    w_temp = _gauss_weight(row["temperature"], profile["t_opt"], profile["t_tol"])
    w_hum  = _gauss_weight(row["humidity"],    profile["h_opt"], profile["h_tol"])
    w_ph   = _gauss_weight(row["ph"],          profile["ph_opt"], profile["ph_tol"])
    w_rain = _gauss_weight(row["rainfall"],    profile["rain_opt"], profile["rain_w"] * 80)

    # NPK plays a smaller, indirect role — under/over-fertilised plants are more
    # susceptible. We use distance from the crop's N optimum.
    n_dist = abs(row["N"] - CROP_N_OPTIMUMS[profile["crop"]]) / 80
    nutrient_penalty = np.clip(n_dist * 12, 0, 12)

    # Linear blend of weighted drivers + base risk + nutrient penalty + noise.
    score = (
        profile["base"]
        + 35 * w_temp
        + 30 * w_hum
        + 10 * w_ph
        + 25 * w_rain * profile["rain_w"]
        + nutrient_penalty
        + np.random.normal(0, 4)
    )
    return float(np.clip(score, 0, 100))


def sample_row(rng: np.random.Generator) -> dict:
    row = {}
    for name, (lo, hi) in FEATURE_RANGES.items():
        row[name] = float(rng.uniform(lo, hi))
    return row


def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    rows = []
    for disease, profile in DISEASE_PROFILES.items():
        for _ in range(N_SAMPLES_PER_DISEASE):
            r = sample_row(rng)
            r["crop"] = profile["crop"]
            r["disease"] = disease
            r["healthy"] = int(profile.get("healthy", False))
            r["risk_score"] = compute_risk(profile, r)
            # Derived ordinal label: Low (<35), Medium (35-65), High (>65)
            if r["risk_score"] < 35:
                r["risk_category"] = "Low"
            elif r["risk_score"] < 65:
                r["risk_category"] = "Medium"
            else:
                r["risk_category"] = "High"
            rows.append(r)
    df = pd.DataFrame(rows)
    cols = ["crop", "disease", "healthy",
            "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
            "risk_score", "risk_category"]
    return df[cols]


def main() -> None:
    df = build_dataset()
    out_csv = OUT_DIR / "plant_disease_risk_dataset.csv"
    df.to_csv(out_csv, index=False)

    meta = {
        "rows": int(len(df)),
        "disease_classes": sorted(df["disease"].unique().tolist()),
        "crops": sorted(df["crop"].unique().tolist()),
        "feature_columns": [
            "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
            "disease",  # categorical, one-hot encoded in training
        ],
        "target_regression": "risk_score",
        "target_classification": "risk_category",
        "sources": [
            "Kaggle 'Crop Recommendation Dataset' (Atharva Ingle) — schema/ranges",
            "Agrios, Plant Pathology 5th ed. — disease-condition windows",
            "PlantVillage class catalogue — disease labels"
        ],
    }
    with open(OUT_DIR / "dataset_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(df):,} rows to {out_csv}")
    print(df.head().to_string(index=False))
    print("\nClass balance (risk_category):")
    print(df["risk_category"].value_counts())

    # ----- PlantVillage download instructions ---------------------------------
    plantvillage_readme = OUT_DIR / "PLANTVILLAGE_README.md"
    plantvillage_readme.write_text(
        "# PlantVillage download instructions\n\n"
        "The CV block uses the [PlantVillage dataset](https://www.kaggle.com/"
        "datasets/abdallahalidev/plantvillage-dataset).\n\n"
        "1. Install the Kaggle CLI:  `pip install kaggle`\n"
        "2. Place your `~/.kaggle/kaggle.json` API token.\n"
        "3. Run:\n\n"
        "    ```bash\n"
        "    mkdir -p data/plantvillage\n"
        "    cd data/plantvillage\n"
        "    kaggle datasets download -d abdallahalidev/plantvillage-dataset\n"
        "    unzip plantvillage-dataset.zip\n"
        "    ```\n\n"
        "4. The training script (`train_cv_model.py`) expects the directory\n"
        "   layout `data/plantvillage/color/<class_name>/*.jpg`.\n"
        "5. We use the 15 classes listed in `data/dataset_metadata.json`.\n"
    )
    print(f"\nWrote {plantvillage_readme}")


if __name__ == "__main__":
    main()
