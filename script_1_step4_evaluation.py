"""
script_1_step4_evaluation.py
Standalone Step 4 evaluation script for the ChestX-ray14 deepfake detection pipeline.

Run this AFTER script_1_chestxray14.py has completed Steps 0-3.
It loads the pre-trained detector and pre-generated test images, then produces
the confusion matrix and full metrics report WITHOUT re-running any prior steps.

Expected pre-existing files:
  ./work_chestxray14/checkpoints/generator_train_only.pt
  ./work_chestxray14/models/best_detector.pt
  ./work_chestxray14/datasets/test/          (real test images)
  ./work_chestxray14/generated_fakes/test/   (GAN-generated fake test images)

Output files (written to OUTPUT_DIR, default "."):
  confusion_matrix_chestxray14.csv
  confusion_matrix_chestxray14.png
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import sys
import csv
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# CONSTANTS  (must match script_1_chestxray14.py)
# ============================================================
DATASET_NAME  = "chestxray14"
IMG_SIZE_DET  = 256
BATCH_SIZE    = 32
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR    = "."
WORK_DIR      = "./work_chestxray14"

# Derived paths
TEST_DIR      = os.path.join(WORK_DIR, "datasets", "test")
FAKES_TEST    = os.path.join(WORK_DIR, "generated_fakes", "test")
MODELS_DIR    = os.path.join(WORK_DIR, "models")
CKPT_DIR      = os.path.join(WORK_DIR, "checkpoints")

DETECTOR_PATH  = os.path.join(MODELS_DIR, "best_detector.pt")
GENERATOR_PATH = os.path.join(CKPT_DIR,   "generator_train_only.pt")

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================
print("=" * 70)
print("STEP 4 (standalone): FINAL EVALUATION ON TEST SET")
print(f"  Device : {DEVICE}")
print("=" * 70)

errors = []

if not os.path.isfile(DETECTOR_PATH):
    errors.append(
        f"[ERROR] Detector model not found: {DETECTOR_PATH}\n"
        "        Run script_1_chestxray14.py first to train and save the detector."
    )

if not os.path.isfile(GENERATOR_PATH):
    errors.append(
        f"[ERROR] Generator checkpoint not found: {GENERATOR_PATH}\n"
        "        Run script_1_chestxray14.py first to train and save the generator."
    )

if not os.path.isdir(TEST_DIR) or not any(
    f.lower().endswith(".png") for f in os.listdir(TEST_DIR)
):
    errors.append(
        f"[ERROR] Real test images not found in: {TEST_DIR}\n"
        "        Run script_1_chestxray14.py (Step 0) first to create the test split."
    )

if not os.path.isdir(FAKES_TEST) or not any(
    f.lower().endswith(".png") for f in os.listdir(FAKES_TEST)
):
    errors.append(
        f"[ERROR] Fake test images not found in: {FAKES_TEST}\n"
        "        Run script_1_chestxray14.py (Step 2) first to generate fake images."
    )

if errors:
    print("\n".join(errors))
    sys.exit(1)

print("✓ All required files found")

# ============================================================
# MODEL DEFINITION  (must match script_1_chestxray14.py)
# ============================================================
class MedicalDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.backbone   = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        f = self.backbone(x)
        f = torch.flatten(f, 1)
        return self.classifier(f)


# ============================================================
# LOAD DETECTOR
# ============================================================
print(f"\n[1/4] Loading detector from {DETECTOR_PATH} ...")
detector = MedicalDeepfakeDetector().to(DEVICE)
detector.load_state_dict(torch.load(DETECTOR_PATH, map_location=DEVICE))
detector.eval()
print("✓ Detector loaded")

# ============================================================
# LOAD TEST IMAGES
# ============================================================
print("\n[2/4] Loading test images ...")

class RealFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, img_size=256):
        self.img_size = img_size
        self.images, self.labels = [], []
        for img_file in sorted(os.listdir(real_dir)):
            if img_file.lower().endswith(".png"):
                self.images.append(os.path.join(real_dir, img_file))
                self.labels.append(0)   # 0 = Real
        for img_file in sorted(os.listdir(fake_dir)):
            if img_file.lower().endswith(".png"):
                self.images.append(os.path.join(fake_dir, img_file))
                self.labels.append(1)   # 1 = Fake

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Could not read image: {self.images[idx]} – using zero tensor")
            return torch.zeros(1, self.img_size, self.img_size), self.labels[idx]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0), self.labels[idx]


test_ds     = RealFakeDataset(TEST_DIR, FAKES_TEST, IMG_SIZE_DET)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=min(os.cpu_count() or 1, 4))

n_real = sum(1 for l in test_ds.labels if l == 0)
n_fake = sum(1 for l in test_ds.labels if l == 1)
print(f"✓ Test set: {n_real} real + {n_fake} fake = {len(test_ds)} total images")

# ============================================================
# RUN INFERENCE
# ============================================================
print("\n[3/4] Running inference ...")
start_time = time.time()

all_preds, all_probs, all_labels = [], [], []

with torch.no_grad():
    for imgs, lbls in tqdm(test_loader, desc="Inference"):
        imgs   = imgs.to(DEVICE)
        logits = detector(imgs)
        probs  = torch.softmax(logits, dim=1)
        preds  = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(lbls.numpy())

elapsed = time.time() - start_time
print(f"✓ Inference complete in {elapsed:.1f}s")

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)

# ============================================================
# COMPUTE METRICS
# ============================================================
accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall    = recall_score(all_labels, all_preds, zero_division=0)
f1        = f1_score(all_labels, all_preds, zero_division=0)
auc       = roc_auc_score(all_labels, all_probs)
cm        = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 70)
print(f"DATASET : {DATASET_NAME}")
print("=" * 70)
print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC       : {auc:.4f}")
print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")

# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================
print("\n[4/4] Saving confusion matrix ...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{DATASET_NAME}.csv")
with open(csv_path, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["", "Predicted Real", "Predicted Fake"])
    writer.writerow(["Actual Real", tn, fp])
    writer.writerow(["Actual Fake", fn, tp])
print(f"✓ CSV  → {csv_path}")

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set(
    xticks=[0, 1], yticks=[0, 1],
    xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
    ylabel="True label", xlabel="Predicted label",
    title=f"Confusion Matrix – {DATASET_NAME}",
)
thresh = cm.max() / 2.0
for r in range(2):
    for c in range(2):
        ax.text(c, r, format(cm[r, c], "d"),
                ha="center", va="center",
                color="white" if cm[r, c] > thresh else "black")
plt.tight_layout()
png_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{DATASET_NAME}.png")
plt.savefig(png_path, dpi=150)
plt.close()
print(f"✓ PNG  → {png_path}")

print("\n" + "=" * 70)
print("Step 4 evaluation complete.")
print("=" * 70)
