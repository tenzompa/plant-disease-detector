---
title: Plant Disease Detector
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
python_version: 3.10
---
---

# 🌿 Plant Disease Detector — Multimodal AI Application

End-to-end AI application that helps small-scale farmers and home gardeners
diagnose plant leaf diseases and receive **context-aware treatment advice**.

The project combines **three AI blocks** in a single workflow:

| Block | Role in the system |
|-------|-------------------|
| **Computer Vision** | Fine-tuned ViT classifies the leaf image into one of 15 PlantVillage disease classes |
| **ML Numeric Data** | sklearn model (Gradient Boosting) estimates the *environmental risk score* for the detected disease using soil + weather features |
| **NLP** | OpenAI LLM extracts environmental conditions from free user text and writes a tailored treatment plan that fuses the CV prediction with the numeric risk |

## How it works
1. The user uploads a leaf photo and writes a short text describing growing
   conditions (e.g. *"Tomatoes in my Zürich greenhouse, ~26 °C, 80 % humidity,
   soil pH 6.4, light rain"*).
2. The **ViT** model predicts the disease class (and the top-3 probabilities).
3. The **LLM** extracts a structured JSON of environmental parameters from
   the text (temperature, humidity, pH, rainfall, soil N/P/K when given).
4. The **sklearn Gradient Boosting** model takes those parameters together
   with the predicted disease and returns a numeric **risk score (0–100)** and
   a **risk category** (Low / Medium / High).
5. The **LLM** generates a natural-language treatment recommendation that
   takes both the disease and the risk score into account.

The same image is also classified by **CLIP zero-shot** and the **OpenAI
vision model** so the user (and the documentation) can compare three vision
models on the same input.

## Required secrets

* `OPENAI_API_KEY` — required for the NLP block

## Files

| File | Purpose |
|------|---------|
| `app.py` | Gradio app that orchestrates CV + NLP + ML numeric |
| `train_cv_model.py` | ViT fine-tuning script for the PlantVillage subset |
| `train_numeric_model.py` | Trains the disease-risk regression model |
| `prepare_data.py` | Downloads / prepares the two source datasets |
| `models/disease_risk_model.pkl` | Trained sklearn Gradient Boosting model |
| `models/numeric_feature_columns.json` | Feature ordering for the numeric model |
| `data/plant_disease_risk_dataset.csv` | Merged numeric dataset (crop conditions + disease risk) |
| `notebooks/01_numeric_eda_and_training.ipynb` | EDA + 2 iterations of model training (numeric block) |
| `notebooks/02_cv_eda_and_training.ipynb` | EDA + transfer learning experiments (CV block) |
| `notebooks/03_nlp_prompt_evaluation.ipynb` | Prompt comparison + LLM evaluation (NLP block) |
| `documentation.md` | Full project documentation (per course template) |
| `example_images/` | Sample leaf photos used in the Gradio demo |
| `screenshots/` | Screenshots of the running app |

## Quick start

```bash
pip install -r requirements.txt
python prepare_data.py                 # downloads / generates datasets
python train_numeric_model.py          # trains and saves the risk model
python train_cv_model.py               # fine-tunes ViT (GPU recommended)
export OPENAI_API_KEY=sk-...
python app.py                          # launches the Gradio UI
```

## Live deployment

* Hugging Face Space: <https://huggingface.co/spaces/tashiten/plant-disease-detector>
* Fine-tuned ViT model: <https://huggingface.co/tashiten/plant-disease-vit>

## Collaborators added to the GitHub repo

* `jasminh` (Jasmin Heierli)
* `bkuehnis` (Benjamin Kühnis)
