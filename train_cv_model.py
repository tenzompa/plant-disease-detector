"""
Train the Vision Transformer (ViT) on a 15-class PlantVillage subset.

Inputs
------
data/plantvillage/color/<class_name>/*.jpg   (downloaded via prepare_data.py)

Outputs
-------
plant-disease-vit/                            — local checkpoint
https://huggingface.co/tashiten/plant-disease-vit    — pushed model
models/cv_training_report.json                — final eval metrics

Device auto-detection: CUDA → Apple MPS → CPU.
Environment variables to keep local Mac training fast:
    MAX_IMAGES_PER_CLASS=300        # default: all images (~ 1700/class)
    NUM_EPOCHS=3                    # default: 5
    BATCH_SIZE=8                    # default: 16
    BASE_MODEL=google/vit-base-patch16-224   # default
    PUSH_TO_HUB=0                   # set to 0 to skip the Hub push

Examples
--------
# Fast local Mac training (≈ 30-45 min on M1/M2)
MAX_IMAGES_PER_CLASS=300 NUM_EPOCHS=3 BATCH_SIZE=8 python train_cv_model.py

# Colab / full GPU training
python train_cv_model.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_score, recall_score, f1_score,
)
from transformers import (
    AutoImageProcessor, Trainer, TrainingArguments, ViTForImageClassification,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_CHECKPOINT = os.getenv("BASE_MODEL", "google/vit-base-patch16-224")
DATA_DIR         = "data/plantvillage/color"   # ImageFolder layout
OUTPUT_DIR       = "./plant-disease-vit"
HF_MODEL_ID      = os.getenv("HF_MODEL_ID", "tashiten/plant-disease-vit")
PUSH_TO_HUB      = os.getenv("PUSH_TO_HUB", "1") == "1"
MAX_IMAGES_PER_CLASS = int(os.getenv("MAX_IMAGES_PER_CLASS", "0"))  # 0 = no cap
NUM_EPOCHS       = int(os.getenv("NUM_EPOCHS", "5"))
BATCH_SIZE       = int(os.getenv("BATCH_SIZE", "16"))

# Restrict to the 15-class subset used by the rest of the project.
TARGET_CLASSES = sorted([
    "Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Bacterial_spot",
    "Potato___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Pepper,_bell___healthy", "Pepper,_bell___Bacterial_spot",
    "Apple___healthy", "Apple___Apple_scab", "Apple___Black_rot",
    "Corn_(maize)___healthy", "Corn_(maize)___Common_rust_",
])

# Mapping from the PlantVillage folder names to our canonical labels.
PLANTVILLAGE_TO_CANONICAL = {
    "Tomato___healthy":               "Tomato___healthy",
    "Tomato___Early_blight":          "Tomato___Early_blight",
    "Tomato___Late_blight":           "Tomato___Late_blight",
    "Tomato___Leaf_Mold":             "Tomato___Leaf_Mold",
    "Tomato___Bacterial_spot":        "Tomato___Bacterial_spot",
    "Potato___healthy":               "Potato___healthy",
    "Potato___Early_blight":          "Potato___Early_blight",
    "Potato___Late_blight":           "Potato___Late_blight",
    "Pepper,_bell___healthy":         "Pepper___healthy",
    "Pepper,_bell___Bacterial_spot":  "Pepper___Bacterial_spot",
    "Apple___healthy":                "Apple___healthy",
    "Apple___Apple_scab":             "Apple___Apple_scab",
    "Apple___Black_rot":              "Apple___Black_rot",
    "Corn_(maize)___healthy":         "Corn___healthy",
    "Corn_(maize)___Common_rust_":    "Corn___Common_rust",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Device auto-detection
    if torch.cuda.is_available():
        print("[device] CUDA available — using GPU")
    elif torch.backends.mps.is_available():
        print("[device] Apple MPS available — using Mac GPU")
    else:
        print("[device] no GPU — training on CPU (this will be slow)")

    dataset = load_dataset("imagefolder", data_dir=DATA_DIR)
    # Keep only the 15 classes of interest.
    raw_labels = dataset["train"].features["label"].names
    keep_idx = {i for i, name in enumerate(raw_labels) if name in TARGET_CLASSES}
    dataset = dataset.filter(lambda ex: ex["label"] in keep_idx)

    # Optional cap to keep local Mac training tractable.
    if MAX_IMAGES_PER_CLASS > 0:
        print(f"[subsample] capping at {MAX_IMAGES_PER_CLASS} images per class")
        from collections import defaultdict
        counts = defaultdict(int)
        keep_rows = []
        for i, ex in enumerate(dataset["train"]):
            if counts[ex["label"]] < MAX_IMAGES_PER_CLASS:
                keep_rows.append(i)
                counts[ex["label"]] += 1
        dataset["train"] = dataset["train"].select(keep_rows)
        print(f"[subsample] dataset reduced to {len(dataset['train'])} images total")

    # Re-index labels using the canonical mapping.
    canonical_names = sorted({PLANTVILLAGE_TO_CANONICAL[raw_labels[i]] for i in keep_idx})
    label2id = {l: i for i, l in enumerate(canonical_names)}
    id2label = {i: l for l, i in label2id.items()}

    def relabel(ex):
        canonical = PLANTVILLAGE_TO_CANONICAL[raw_labels[ex["label"]]]
        ex["label"] = label2id[canonical]
        return ex
    dataset = dataset.map(relabel)

    # train / test split (80 / 20, stratified by label).
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42, stratify_by_column="label")

    processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

    def transform(batch):
        images = [img.convert("RGB") if isinstance(img, Image.Image) else img for img in batch["image"]]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = batch["label"]
        return inputs

    dataset = dataset.with_transform(transform)

    model = ViTForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(canonical_names),
        label2id={l: str(i) for l, i in label2id.items()},
        id2label={str(i): l for i, l in id2label.items()},
        ignore_mismatched_sizes=True,
    )

    def collate(batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels":       torch.tensor([x["labels"] for x in batch]),
        }

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        return {
            "accuracy":  accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="weighted", zero_division=0),
            "recall":    recall_score(labels, preds, average="weighted", zero_division=0),
            "f1":        f1_score(labels, preds, average="weighted", zero_division=0),
        }

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=NUM_EPOCHS,
        learning_rate=3e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=HF_MODEL_ID,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=dataset["train"], eval_dataset=dataset["test"],
        tokenizer=processor, data_collator=collate, compute_metrics=compute_metrics,
    )

    trainer.train()
    final = trainer.evaluate()
    print("Final evaluation:", final)

    # Detailed classification report on test split
    preds_out = trainer.predict(dataset["test"])
    preds = np.argmax(preds_out.predictions, axis=1)
    labels = preds_out.label_ids
    report = classification_report(labels, preds, target_names=canonical_names, digits=3, zero_division=0)
    print(report)

    if PUSH_TO_HUB:
        trainer.push_to_hub()
        processor.push_to_hub(HF_MODEL_ID)
        print(f"Model uploaded to: https://huggingface.co/{HF_MODEL_ID}")
    else:
        print("[push] PUSH_TO_HUB=0 — skipping hub push. "
              f"Local checkpoint saved in {OUTPUT_DIR}/")

    Path("models").mkdir(exist_ok=True)
    with open("models/cv_training_report.json", "w") as f:
        json.dump({
            "final_eval":  {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in final.items()},
            "classes":     canonical_names,
            "model_id":    HF_MODEL_ID,
            "base_model":  MODEL_CHECKPOINT,
            "epochs":      args.num_train_epochs,
            "batch_size":  args.per_device_train_batch_size,
            "lr":          args.learning_rate,
        }, f, indent=2)

if __name__ == "__main__":
    main()
