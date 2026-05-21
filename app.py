"""
Plant Disease Detector — multimodal Gradio app.

Pipeline (executed on every "Diagnose" click):
  1) ViT classifies the uploaded leaf image (custom fine-tune on PlantVillage).
  2) CLIP zero-shot + OpenAI vision classify the same image — model comparison.
  3) LLM #1 extracts structured environmental parameters from the user's text.
  4) sklearn HistGradientBoosting computes the disease-progression risk score
     from (env params + predicted disease).
  5) LLM #2 writes a tailored treatment plan that fuses the disease and the
     risk score, in plain language.

Required Hugging Face Space secret:  OPENAI_API_KEY
"""

from __future__ import annotations

import base64
import inspect
import json
import mimetypes
import os
from pathlib import Path

# Prevent Hugging Face tokenizer fork warning.
# Important: keep this BEFORE importing transformers/pipeline.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from PIL import Image
from transformers import pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CUSTOM_VIT_MODEL = os.getenv("CUSTOM_VIT_MODEL", "tashiten/plant-disease-vit")
CLIP_MODEL_ID    = "openai/clip-vit-large-patch14"
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
NUMERIC_MODEL    = Path("models/disease_risk_model.pkl")
NUMERIC_META     = Path("models/numeric_feature_columns.json")

# Use GPU if available, otherwise CPU.
HF_DEVICE = 0 if torch.cuda.is_available() else -1
print(f"[startup] Hugging Face device: {'cuda:0' if HF_DEVICE == 0 else 'cpu'}")

# Compatibility helpers for different Gradio versions.
# Newer Gradio supports render_children on gr.Tab.
TAB_RENDER_KW = (
    {"render_children": True}
    if "render_children" in inspect.signature(gr.Tab.__init__).parameters
    else {}
)

# Newer Gradio supports height on gr.JSON.
JSON_HEIGHT_KW = (
    {"height": 220}
    if "height" in inspect.signature(gr.JSON.__init__).parameters
    else {}
)

# Cost safety: cap OpenAI calls per process to avoid runaway billing.
# Each "Diagnose" click in the UI uses 3 calls. Default cap = 60 clicks.
OPENAI_CALL_CAP  = int(os.getenv("OPENAI_CALL_CAP", "180"))
_openai_call_count = 0


def _check_openai_budget():
    global _openai_call_count
    if _openai_call_count >= OPENAI_CALL_CAP:
        raise RuntimeError(
            f"OPENAI_CALL_CAP ({OPENAI_CALL_CAP}) reached. "
            "Restart the app or raise the cap to continue."
        )
    _openai_call_count += 1


with open(NUMERIC_META) as f:
    NUMERIC_META_DATA = json.load(f)

NUMERIC_FEATURES   = NUMERIC_META_DATA["numeric_columns"]
ALL_FEATURE_COLS   = NUMERIC_META_DATA["feature_columns"]
KNOWN_DISEASE_LBLS = NUMERIC_META_DATA["disease_classes"]

CROP_N_OPT = {"Tomato": 90, "Potato": 80, "Pepper": 70, "Apple": 50, "Corn": 100}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

print(f"[startup] loading numeric model from {NUMERIC_MODEL}")
risk_model = joblib.load(NUMERIC_MODEL)

print(f"[startup] loading ViT model: {CUSTOM_VIT_MODEL}")
try:
    vit_classifier = pipeline(
        "image-classification",
        model=CUSTOM_VIT_MODEL,
        device=HF_DEVICE,
    )
except Exception as exc:  # pragma: no cover — happens before HF model is pushed
    print(f"[startup] WARNING: could not load custom ViT ({exc}). Falling back to base ViT.")
    vit_classifier = pipeline(
        "image-classification",
        model="google/vit-base-patch16-224",
        device=HF_DEVICE,
    )

print(f"[startup] loading CLIP: {CLIP_MODEL_ID}")
clip_detector = pipeline(
    task="zero-shot-image-classification",
    model=CLIP_MODEL_ID,
    device=HF_DEVICE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "OPENAI_API_KEY not set. Add it under Settings → Variables and secrets."
        )
    return OpenAI(api_key=key)


def _image_to_data_url(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(Path(image_path).read_bytes()).decode()}"


def _parse_json(text: str) -> dict:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")
        ).strip()
    return json.loads(cleaned)


def _short_label(raw: str) -> str:
    """Coerce a raw ViT label like 'Tomato___Late_blight' to a friendly form."""
    return raw.replace("___", " — ").replace("_", " ")


def _normalise_to_known(label: str) -> str:
    """Map a ViT label to one of the 15 known disease classes."""
    if label in KNOWN_DISEASE_LBLS:
        return label

    # Try a fuzzy contains lookup with different separator conventions.
    label_norm = label.replace(",_bell", "").replace("(maize)", "").replace("_", "")

    for known in KNOWN_DISEASE_LBLS:
        if known.replace("_", "") == label_norm:
            return known

    # Last fallback: try matching crop prefix + healthy.
    for known in KNOWN_DISEASE_LBLS:
        if (
            known.split("___")[0].lower() in label.lower()
            and "healthy" in known.lower()
            and "healthy" in label.lower()
        ):
            return known

    # Leave as-is; the numeric model will treat it as unknown OHE.
    return label


# ---------------------------------------------------------------------------
# Vision block
# ---------------------------------------------------------------------------

def cv_predict(image_path: str) -> dict:
    """Run the fine-tuned ViT and return the top-3 (label, score) pairs."""
    img = Image.open(image_path).convert("RGB")
    raw = vit_classifier(img, top_k=3)
    return {r["label"]: float(r["score"]) for r in raw}


def clip_predict(image_path: str) -> dict:
    candidates = [_short_label(d) for d in KNOWN_DISEASE_LBLS]
    img = Image.open(image_path).convert("RGB")
    raw = clip_detector(img, candidate_labels=candidates)[:3]
    return {r["label"]: float(r["score"]) for r in raw}


def openai_vision_predict(image_path: str) -> dict:
    try:
        client = _client()
        _check_openai_budget()
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc)}

    prompt = (
        "You are a plant pathologist. Classify the leaf in the image into "
        "exactly one of these PlantVillage classes (use the underscores): "
        + ", ".join(KNOWN_DISEASE_LBLS)
        + ". Return ONLY this JSON: "
        + '{"top1":"...","top3":["...","...","..."],"reason":"short note"}'
    )

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": _image_to_data_url(image_path)},
                ],
            }
        ],
        temperature=0,
        max_output_tokens=200,
    )

    try:
        return _parse_json(resp.output_text)
    except Exception:
        return {"raw": resp.output_text}


# ---------------------------------------------------------------------------
# NLP extraction
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = (
    "You are an assistant that extracts plant growing conditions from a user's "
    "free text. Respond ONLY with a JSON object. The JSON must contain these "
    "seven keys with numeric values (use a reasonable default if the user did "
    "not specify a particular field): "
    "{\"N\": ppm, \"P\": ppm, \"K\": ppm, \"temperature\": Celsius, "
    "\"humidity\": percent, \"ph\": float, \"rainfall\": mm_per_month}. "
    "Defaults if missing: N=70, P=50, K=50, temperature=22, humidity=70, ph=6.5, rainfall=80. "
    "Return only JSON, no markdown."
)


def extract_conditions(user_text: str) -> dict:
    client = _client()
    _check_openai_budget()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": user_text},
        ],
    )

    parsed = _parse_json(resp.choices[0].message.content)

    # Coerce + clip to safe ranges.
    out = {}
    for k, default in [
        ("N", 70),
        ("P", 50),
        ("K", 50),
        ("temperature", 22),
        ("humidity", 70),
        ("ph", 6.5),
        ("rainfall", 80),
    ]:
        try:
            out[k] = float(parsed.get(k, default))
        except Exception:
            out[k] = default

    out["humidity"] = float(np.clip(out["humidity"], 0, 100))
    out["ph"]       = float(np.clip(out["ph"], 3.5, 9.5))

    return out


# ---------------------------------------------------------------------------
# Numeric risk
# ---------------------------------------------------------------------------

def compute_risk(env: dict, disease_label: str) -> tuple[float, str]:
    disease = _normalise_to_known(disease_label)
    crop = disease.split("___")[0] if "___" in disease else "Tomato"

    thi = env["temperature"] - (0.55 - 0.0055 * env["humidity"]) * (env["temperature"] - 14.5)
    n_balance = env["N"] - CROP_N_OPT.get(crop, 80)

    row = {
        **env,
        "temp_humidity_index": thi,
        "n_balance": n_balance,
        "disease": disease,
    }

    X = pd.DataFrame([row])[ALL_FEATURE_COLS]

    score = float(risk_model.predict(X)[0])
    score = float(np.clip(score, 0, 100))

    if score < 35:
        cat = "Low"
    elif score < 65:
        cat = "Medium"
    else:
        cat = "High"

    return score, cat


# ---------------------------------------------------------------------------
# NLP treatment plan
# ---------------------------------------------------------------------------

TREATMENT_SYSTEM = (
    "You are a friendly agronomy assistant. The user just had a leaf photo "
    "diagnosed. The disease class, environmental conditions and a numeric "
    "risk score (0-100) are given. Write a short treatment plan (English, "
    "120-180 words). Cover: (1) one-sentence disease description; (2) three "
    "concrete actions tailored to the risk level (urgent if High, preventive "
    "if Low); (3) one note about how the current weather/soil makes the "
    "disease more or less aggressive; (4) one safety/uncertainty disclaimer. "
    "Return JSON only: {\"summary\": \"...\"}."
)


def treatment_plan(disease: str, env: dict, score: float, category: str) -> str:
    if disease.endswith("healthy"):
        prompt_extra = "The leaf is healthy. Give a short prevention plan instead of a treatment plan."
    else:
        prompt_extra = ""

    client = _client()
    _check_openai_budget()

    user = (
        f"Disease: {disease}\n"
        f"Environment: {json.dumps(env)}\n"
        f"Risk score: {score:.1f}/100  (category: {category})\n"
        f"{prompt_extra}"
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": TREATMENT_SYSTEM},
            {"role": "user", "content": user},
        ],
    )

    return _parse_json(resp.choices[0].message.content).get("summary", "")


# ---------------------------------------------------------------------------
# Diagnosis headline helper
# ---------------------------------------------------------------------------

_RISK_COLORS = {"Low": "#2e7d32", "Medium": "#ed6c02", "High": "#c62828"}


def _diagnosis_markdown(disease_label: str, risk: float, category: str) -> str:
    """Compact, prominent summary shown at the top of the Diagnosis tab."""
    pretty = _short_label(disease_label)
    color = _RISK_COLORS.get(category, "#555")

    return (
        f"### 🩺 Detected disease: **{pretty}**\n"
        f"<span style=\"background:{color};color:white;padding:4px 10px;"
        f"border-radius:6px;font-weight:600;\">Risk {risk:.1f} / 100 — {category}</span>"
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def run_pipeline(image_path, user_text):
    if image_path is None:
        return (
            "⚠️ Please upload a leaf image to start.",
            {},
            {},
            {"status": "No image uploaded."},
            {"status": "No image uploaded."},
            0.0,
            "Low",
            "",
        )

    if not user_text or not user_text.strip():
        user_text = "No environment specified."

    # 1) Vision predictions
    vit_pred = cv_predict(image_path)
    clip_pred = clip_predict(image_path)
    openai_pred = openai_vision_predict(image_path)

    top_disease = max(vit_pred, key=vit_pred.get)

    # 2) NLP extraction
    try:
        env = extract_conditions(user_text)
    except Exception as exc:
        env = {"error": str(exc)}
        return (
            "",
            vit_pred,
            clip_pred,
            openai_pred,
            env,
            0.0,
            "Low",
            f"❌ Extraction error: {exc}",
        )

    # 3) Numeric risk
    try:
        risk, cat = compute_risk(env, top_disease)
    except Exception as exc:
        return (
            "",
            vit_pred,
            clip_pred,
            openai_pred,
            env,
            0.0,
            "Low",
            f"❌ Risk error: {exc}",
        )

    # 4) Treatment plan
    try:
        plan = treatment_plan(top_disease, env, risk, cat)
    except Exception as exc:
        plan = f"❌ Treatment error: {exc}"

    headline = _diagnosis_markdown(top_disease, round(risk, 1), cat)

    return (
        headline,
        vit_pred,
        clip_pred,
        openai_pred,
        env,
        round(risk, 1),
        cat,
        plan,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLES = [
    [
        "example_images/tomato_late_blight.jpg",
        "Tomatoes outdoors in Zurich, ~18°C, 92% humidity, light rain almost every day, soil pH 6.0, low fertiliser.",
    ],
    [
        "example_images/tomato_healthy.jpg",
        "Greenhouse tomato in Winterthur, 25°C, 65% humidity, soil pH 6.5, balanced fertilising, no rain.",
    ],
    [
        "example_images/potato_early_blight.jpg",
        "Potato field, warm sunny week, 27°C, 80% humidity, pH 5.9, last rainfall 130mm, N around 100 ppm.",
    ],
    [
        "example_images/apple_scab.jpg",
        "Apple tree in Spring, 17°C, 95% humidity, pH 6.1, frequent rain (~200mm/month), moderate fertilising.",
    ],
    [
        "example_images/corn_common_rust.jpg",
        "Corn field, 22°C, 88% humidity, pH 6.2, 130mm rainfall, N 110 ppm.",
    ],
]

CUSTOM_CSS = """
/* Full page */
body {
    margin: 0;
}

/* Let Gradio use the full browser width */
.gradio-container {
    max-width: none !important;
    width: 100% !important;
}

/* Center the actual app content */
#app-shell {
    max-width: 1420px;
    width: 100%;
    margin: 0 auto !important;
    padding: 32px 32px 48px 32px;
    box-sizing: border-box;
}

/* Keep the title area nicely aligned */
#app-header {
    margin-bottom: 24px;
}

/* Prevent JSON areas from expanding strangely in tabs */
.json-holder {
    max-height: 260px;
    overflow: auto;
}
"""

with gr.Blocks(title="Plant Disease Detector") as demo:

    gr.Markdown(
        """
        # 🌿 Plant Disease Detector
        Multimodal AI for plant-leaf diagnosis — **Computer Vision** (custom ViT) +
        **ML Numeric Risk** (Gradient Boosting on weather & soil) +
        **NLP** (LLM extraction + risk-aware treatment plan).

        Upload a leaf photo, describe your growing conditions, and get a tailored treatment plan.
        """
    )

    with gr.Row(equal_height=False):
        # ----------------------- INPUT COLUMN -----------------------
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 📥 Input")

            image_in = gr.Image(
                type="filepath",
                label="Leaf photo",
                height=320,
            )

            text_in = gr.Textbox(
                label="Growing conditions (free text)",
                lines=5,
                placeholder=(
                    "e.g. Tomato in Zurich, ~22°C, 80% humidity, "
                    "pH 6.4, light rain, N around 80 ppm"
                ),
            )

            btn = gr.Button("🔍 Diagnose", variant="primary", size="lg")

            gr.Markdown(
                "<sub>The pipeline runs CV ➜ NLP extraction ➜ numeric risk ➜ LLM treatment plan. "
                "Takes ~10–20 s per diagnosis.</sub>"
            )

        # ----------------------- OUTPUT COLUMN ----------------------
        with gr.Column(scale=2, min_width=480):
            with gr.Tabs():

                # ===== Tab 1 — Main diagnosis =====
                with gr.Tab("🩺 Diagnosis", **TAB_RENDER_KW):
                    headline_md = gr.Markdown(
                        "*Click **Diagnose** to start. Results will appear here.*"
                    )

                    with gr.Row():
                        risk_out = gr.Number(
                            label="Disease progression risk (0-100)",
                            value=0.0,
                        )
                        cat_out = gr.Textbox(
                            label="Risk category",
                            value="",
                        )

                    plan_out = gr.Textbox(
                        label="🌱 Treatment plan",
                        lines=12,
                        value="",
                    )

                # ===== Tab 2 — Vision model comparison =====
                with gr.Tab("🔬 Vision model comparison", **TAB_RENDER_KW):
                    gr.Markdown(
                        "Three independent vision models classify the same leaf. "
                        "Disagreement is a useful signal — when models agree, confidence is high."
                    )

                    with gr.Row():
                        vit_out = gr.Label(
                            label="Custom ViT (fine-tuned on PlantVillage)",
                            num_top_classes=3,
                            value={},
                        )

                        clip_out = gr.Label(
                            label="CLIP zero-shot (openai/clip-vit-large-patch14)",
                            num_top_classes=3,
                            value={},
                        )

                    oai_out = gr.JSON(
                        label="OpenAI vision (gpt-4o-mini)",
                        value={"status": "Waiting for diagnosis..."},
                        elem_classes=["json-holder"],
                        **JSON_HEIGHT_KW,
                    )

                # ===== Tab 3 — Pipeline details =====
                with gr.Tab("⚙️ Pipeline details", **TAB_RENDER_KW):
                    gr.Markdown(
                        "Structured environmental conditions extracted by the LLM "
                        "from your free-text description. These are the features the "
                        "numeric risk model consumes."
                    )

                    env_out = gr.JSON(
                        label="Extracted environmental conditions",
                        value={"status": "Waiting for diagnosis..."},
                        elem_classes=["json-holder"],
                        **JSON_HEIGHT_KW,
                    )

    gr.Markdown("### 🧪 Example scenarios")

    gr.Examples(
        EXAMPLES,
        inputs=[image_in, text_in],
        label="Click an example to autofill",
    )

    gr.Markdown(
        "<sub>Custom ViT model: "
        "<a href='https://huggingface.co/tashiten/plant-disease-vit' target='_blank'>"
        "tashiten/plant-disease-vit</a>. "
        "Built for the ZHAW *AI Applications* course (2026).</sub>"
    )

    btn.click(
        run_pipeline,
        inputs=[image_in, text_in],
        outputs=[
            headline_md,
            vit_out,
            clip_out,
            oai_out,
            env_out,
            risk_out,
            cat_out,
            plan_out,
        ],
    )


if __name__ == "__main__":
    try:
        demo.launch(
            theme=gr.themes.Soft(primary_hue="green", secondary_hue="amber"),
            css=CUSTOM_CSS,
            show_error=True,
        )
    except TypeError:
        # Older Gradio: theme/css/show_error handling can differ.
        demo.launch(show_error=True)