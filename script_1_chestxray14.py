"""
script_1_chestxray14.py
Leakage-free deepfake detection pipeline applied to ChestX-ray14 dataset.

Dataset: NIH ChestX-ray14
  - Images: PNG files in images/ folder (or images_001/, images_002/, etc.)
  - Metadata: Data_Entry_2017.csv (Image Index, Finding Labels, ...)

Before running, set DATA_PATH to the root of your ChestX-ray14 download.
Expected layout:
  DATA_PATH/
    images/          (or images_001/, images_002/, ... sub-folders)
    Data_Entry_2017.csv   (optional – used only for stratification hint)

Output files (written next to this script or in OUTPUT_DIR):
  confusion_matrix_chestxray14.csv
  confusion_matrix_chestxray14.png
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import sys
import random
import csv

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# USER CONFIGURATION  –  edit these paths
# ============================================================
DATA_PATH   = os.environ.get("CHESTXRAY14_PATH", "./chestxray14_data")
OUTPUT_DIR  = "."          # where confusion matrix files are saved
WORK_DIR    = "./work_chestxray14"   # working directory for split images / fakes

# ============================================================
# PIPELINE CONSTANTS  (do NOT change)
# ============================================================
DATASET_NAME = "chestxray14"
IMG_SIZE_GAN  = 64
IMG_SIZE_DET  = 256
LATENT_DIM    = 100
GAN_EPOCHS    = 100
DET_EPOCHS    = 50
BATCH_SIZE    = 32
LR_GAN        = 0.0002
LR_DET        = 0.001
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE  = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Derived paths
DATASET_DIR   = os.path.join(WORK_DIR, "datasets")
TRAIN_DIR     = os.path.join(DATASET_DIR, "train")
VAL_DIR       = os.path.join(DATASET_DIR, "val")
TEST_DIR      = os.path.join(DATASET_DIR, "test")
FAKES_DIR     = os.path.join(WORK_DIR, "generated_fakes")
FAKES_VAL     = os.path.join(FAKES_DIR, "val")
FAKES_TEST    = os.path.join(FAKES_DIR, "test")
MODELS_DIR    = os.path.join(WORK_DIR, "models")
CKPT_DIR      = os.path.join(WORK_DIR, "checkpoints")

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR, FAKES_VAL, FAKES_TEST, MODELS_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# STEP 0 – Load & split ChestX-ray14 images (60/20/20)
# ============================================================
print("=" * 70)
print("STEP 0: DATA LOADING & SPLITTING (ChestX-ray14)")
print("=" * 70)

def collect_chestxray14_images(data_path):
    """Collect all .png image paths from ChestX-ray14 directory tree."""
    candidates = []
    # Support flat images/ or numbered sub-folders images_001/ etc.
    for root, _dirs, files in os.walk(data_path):
        for fname in files:
            if fname.lower().endswith(".png"):
                candidates.append(os.path.join(root, fname))
    return sorted(candidates)

all_paths = collect_chestxray14_images(DATA_PATH)

if len(all_paths) == 0:
    sys.exit(
        f"[ERROR] No PNG images found under DATA_PATH='{DATA_PATH}'.\n"
        "Please set DATA_PATH to the root of your ChestX-ray14 download."
    )

print(f"Found {len(all_paths)} images")

indices = np.arange(len(all_paths))
train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=RANDOM_STATE)
val_idx,   test_idx = train_test_split(temp_idx, test_size=0.5, random_state=RANDOM_STATE)

print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

def copy_and_resize(paths, dest_dir, img_size, desc):
    os.makedirs(dest_dir, exist_ok=True)
    for i, p in enumerate(tqdm(paths, desc=desc)):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        out_name = f"{i:06d}_{os.path.basename(p)}"
        cv2.imwrite(os.path.join(dest_dir, out_name), img)

copy_and_resize([all_paths[i] for i in train_idx], TRAIN_DIR, IMG_SIZE_GAN, "Copy train")
copy_and_resize([all_paths[i] for i in val_idx],   VAL_DIR,   IMG_SIZE_GAN, "Copy val")
copy_and_resize([all_paths[i] for i in test_idx],  TEST_DIR,  IMG_SIZE_GAN, "Copy test")

# Verify no overlap
train_files = set(os.listdir(TRAIN_DIR))
val_files   = set(os.listdir(VAL_DIR))
test_files  = set(os.listdir(TEST_DIR))
assert len(train_files & val_files) == 0,   "Train/Val overlap!"
assert len(train_files & test_files) == 0,  "Train/Test overlap!"
assert len(val_files   & test_files) == 0,  "Val/Test overlap!"
print("✓ No overlap verified")

# ============================================================
# GAN ARCHITECTURE  (exact copy from trial2.ipynb)
# ============================================================
class ImprovedGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_size=64):
        super().__init__()
        self.fc    = nn.Linear(latent_dim, 512 * 4 * 4)
        self.bn_fc = nn.BatchNorm1d(512 * 4 * 4)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,  64, 4, 2, 1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d( 64,  32, 4, 2, 1, bias=False), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32, 1, 3, 1, 1), nn.Tanh(),
        )
    def forward(self, z):
        x = self.fc(z)
        x = self.bn_fc(x)
        x = x.view(x.size(0), 512, 4, 4)
        return self.model(x)


class ImprovedDiscriminator(nn.Module):
    def __init__(self, img_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,  32, 4, 2, 1),                              nn.LeakyReLU(0.2, True),
            nn.Conv2d(32,  64, 4, 2, 1, bias=False), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128,256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, 1)
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MedicalImageDataset(Dataset):
    def __init__(self, img_dir, img_size=64):
        self.img_dir   = img_dir
        self.img_size  = img_size
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.img_files[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return torch.zeros(1, self.img_size, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return torch.from_numpy(img).unsqueeze(0)

# ============================================================
# STEP 1 – Train GAN on train split only
# ============================================================
print("=" * 70)
print("STEP 1: GAN TRAINING (train split only)")
print("=" * 70)

generator     = ImprovedGenerator(LATENT_DIM, IMG_SIZE_GAN).to(DEVICE)
discriminator = ImprovedDiscriminator(IMG_SIZE_GAN).to(DEVICE)

criterion   = nn.BCEWithLogitsLoss()
opt_g = optim.Adam(generator.parameters(),     lr=LR_GAN, betas=(0.5, 0.999))
opt_d = optim.Adam(discriminator.parameters(), lr=LR_GAN, betas=(0.5, 0.999))
sched_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=GAN_EPOCHS)
sched_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=GAN_EPOCHS)

train_ds     = MedicalImageDataset(TRAIN_DIR, IMG_SIZE_GAN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
print(f"Training GAN on {len(train_ds)} images  |  device={DEVICE}")

for epoch in range(GAN_EPOCHS):
    pbar = tqdm(train_loader, desc=f"GAN Epoch {epoch+1}/{GAN_EPOCHS}", leave=False)
    for real_imgs in pbar:
        real_imgs  = real_imgs.to(DEVICE)
        bs         = real_imgs.size(0)

        opt_d.zero_grad()
        d_real = discriminator(real_imgs)
        loss_d_real = criterion(d_real, torch.ones(bs, 1, device=DEVICE))
        z = torch.randn(bs, LATENT_DIM, device=DEVICE)
        fake_imgs = generator(z)
        d_fake = discriminator(fake_imgs.detach())
        loss_d_fake = criterion(d_fake, torch.zeros(bs, 1, device=DEVICE))
        (loss_d_real + loss_d_fake).backward()
        opt_d.step()

        opt_g.zero_grad()
        z = torch.randn(bs, LATENT_DIM, device=DEVICE)
        fake_imgs = generator(z)
        g_out = discriminator(fake_imgs)
        loss_g = criterion(g_out, torch.ones(bs, 1, device=DEVICE))
        loss_g.backward()
        opt_g.step()

    sched_g.step()
    sched_d.step()

gen_path = os.path.join(CKPT_DIR, "generator_train_only.pt")
torch.save(generator.state_dict(), gen_path)
print(f"✓ Generator saved → {gen_path}")

# ============================================================
# STEP 2 – Generate fakes for val and test separately
# ============================================================
print("=" * 70)
print("STEP 2: GENERATE FAKE IMAGES (val & test)")
print("=" * 70)

generator.eval()

def generate_fakes(dest_dir, count, prefix):
    os.makedirs(dest_dir, exist_ok=True)
    with torch.no_grad():
        for i in tqdm(range(0, count, BATCH_SIZE), desc=f"Fakes {prefix}"):
            bs = min(BATCH_SIZE, count - i)
            z  = torch.randn(bs, LATENT_DIM, device=DEVICE)
            imgs = generator(z)
            for j in range(bs):
                img = imgs[j].cpu().numpy().squeeze()
                img = ((img + 1) / 2 * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(dest_dir, f"fake_{prefix}_{i+j:05d}.png"), img)

val_count  = len(os.listdir(VAL_DIR))
test_count = len(os.listdir(TEST_DIR))
generate_fakes(FAKES_VAL,  val_count,  "val")
generate_fakes(FAKES_TEST, test_count, "test")
print(f"✓ Generated {val_count} val fakes and {test_count} test fakes")

# ============================================================
# DETECTOR DATASET
# ============================================================
class RealFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, img_size=256):
        self.img_size = img_size
        self.images, self.labels = [], []
        for img_file in sorted(os.listdir(real_dir)):
            if img_file.lower().endswith(".png"):
                self.images.append(os.path.join(real_dir, img_file))
                self.labels.append(0)
        for img_file in sorted(os.listdir(fake_dir)):
            if img_file.lower().endswith(".png"):
                self.images.append(os.path.join(fake_dir, img_file))
                self.labels.append(1)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            return torch.zeros(1, self.img_size, self.img_size), self.labels[idx]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0), self.labels[idx]


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
# STEP 3 – Train detector on val split only
# ============================================================
print("=" * 70)
print("STEP 3: DETECTOR TRAINING (val split only)")
print("=" * 70)

val_ds  = RealFakeDataset(VAL_DIR, FAKES_VAL, IMG_SIZE_DET)
indices = np.arange(len(val_ds))
labels_arr = np.array(val_ds.labels)
tr_idx, iv_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE, stratify=labels_arr)

tr_loader = DataLoader(torch.utils.data.Subset(val_ds, tr_idx),  batch_size=BATCH_SIZE, shuffle=True)
iv_loader = DataLoader(torch.utils.data.Subset(val_ds, iv_idx),  batch_size=BATCH_SIZE)

detector  = MedicalDeepfakeDetector().to(DEVICE)
criterion_det = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.2]).to(DEVICE))
opt_det   = optim.Adam(detector.parameters(), lr=LR_DET, weight_decay=1e-5)
sched_det = optim.lr_scheduler.ReduceLROnPlateau(opt_det, mode="max", factor=0.5, patience=3)

best_val_auc  = 0.0
patience_cnt  = 0

for epoch in range(DET_EPOCHS):
    detector.train()
    for imgs, lbls in tqdm(tr_loader, desc=f"Det Epoch {epoch+1}/{DET_EPOCHS}", leave=False):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        logits = detector(imgs)
        loss   = criterion_det(logits, lbls)
        opt_det.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
        opt_det.step()

    detector.eval()
    iv_preds, iv_labels, iv_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in iv_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            logits = detector(imgs)
            probs  = torch.softmax(logits, 1)
            preds  = torch.argmax(logits, 1)
            iv_preds.extend(preds.cpu().numpy())
            iv_labels.extend(lbls.cpu().numpy())
            iv_probs.extend(probs[:, 1].cpu().numpy())

    val_auc = roc_auc_score(iv_labels, iv_probs)
    sched_det.step(val_auc)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_cnt = 0
        torch.save(detector.state_dict(), os.path.join(MODELS_DIR, "best_detector.pt"))
    else:
        patience_cnt += 1
        if patience_cnt >= 5:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"✓ Best val AUC: {best_val_auc:.4f}")

# ============================================================
# STEP 4 – Evaluate on test set
# ============================================================
print("=" * 70)
print("STEP 4: FINAL EVALUATION ON TEST SET")
print("=" * 70)

det_path = os.path.join(MODELS_DIR, "best_detector.pt")
if not os.path.isfile(det_path):
    sys.exit(f"[ERROR] Detector model not found: {det_path}")
detector.load_state_dict(torch.load(det_path, map_location=DEVICE))
detector.eval()
print(f"✓ Detector loaded from {det_path}")

# --- Load real test images ---
print("Loading real test images...")
test_images, test_labels = [], []
skipped_real = 0
real_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".png")])
for f in real_files:
    fpath = os.path.join(TEST_DIR, f)
    try:
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [WARN] Could not read real image: {f}")
            skipped_real += 1
            continue
        img = cv2.resize(img, (IMG_SIZE_DET, IMG_SIZE_DET))
        img = img.astype(np.float32) / 255.0
        test_images.append(torch.from_numpy(img).unsqueeze(0))
        test_labels.append(0)
    except Exception as e:
        print(f"  [WARN] Error loading real image {f}: {e}")
        skipped_real += 1
print(f"  Loaded {len(test_labels)} real images (skipped {skipped_real})")

# --- Load fake test images ---
print("Loading fake test images...")
skipped_fake = 0
fake_files = sorted([f for f in os.listdir(FAKES_TEST) if f.lower().endswith(".png")])
for f in fake_files:
    fpath = os.path.join(FAKES_TEST, f)
    try:
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [WARN] Could not read fake image: {f}")
            skipped_fake += 1
            continue
        img = cv2.resize(img, (IMG_SIZE_DET, IMG_SIZE_DET))
        img = img.astype(np.float32) / 255.0
        test_images.append(torch.from_numpy(img).unsqueeze(0))
        test_labels.append(1)
    except Exception as e:
        print(f"  [WARN] Error loading fake image {f}: {e}")
        skipped_fake += 1
n_real = test_labels.count(0)
n_fake = test_labels.count(1)
print(f"  Loaded {n_fake} fake images (skipped {skipped_fake})")
print(f"Total test images: {len(test_images)} ({n_real} real + {n_fake} fake)")

if len(test_images) == 0:
    sys.exit("[ERROR] No test images were loaded. Cannot run evaluation.")

# --- Run inference ---
print("Running inference on test set...")
all_preds, all_probs, all_labels = [], [], []
n_batches = (len(test_images) + BATCH_SIZE - 1) // BATCH_SIZE
with torch.no_grad():
    for i in tqdm(range(0, len(test_images), BATCH_SIZE), desc="Inference", total=n_batches):
        try:
            batch_tensors = test_images[i:i + BATCH_SIZE]
            batch_labels  = test_labels[i:i + BATCH_SIZE]
            batch = torch.stack(batch_tensors).to(DEVICE)
            logits = detector(batch)
            probs  = torch.softmax(logits, 1)
            preds  = torch.argmax(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(batch_labels)
        except Exception as e:
            print(f"  [WARN] Batch {i // BATCH_SIZE + 1}/{n_batches} failed: {e}")
            continue

if len(all_preds) == 0:
    sys.exit("[ERROR] Inference produced no predictions. Cannot compute metrics.")

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)
print(f"✓ Inference complete: {len(all_preds)} predictions")

# --- Compute metrics ---
print("Computing metrics...")
accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall    = recall_score(all_labels, all_preds, zero_division=0)
f1        = f1_score(all_labels, all_preds, zero_division=0)
try:
    auc = roc_auc_score(all_labels, all_probs)
except ValueError as e:
    print(f"  [WARN] AUC could not be computed: {e}")
    auc = float("nan")

print("\n" + "=" * 70)
print(f"DATASET : {DATASET_NAME}")
print("=" * 70)
print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC       : {auc:.4f}")

# --- Confusion matrix ---
print("Generating confusion matrix...")
try:
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    else:
        print(f"  [WARN] Unexpected confusion matrix shape: {cm.shape}")
        tn = fp = fn = tp = 0
except Exception as e:
    print(f"  [ERROR] Confusion matrix calculation failed: {e}")
    cm = np.zeros((2, 2), dtype=int)
    tn = fp = fn = tp = 0

# Save confusion matrix CSV
csv_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{DATASET_NAME}.csv")
try:
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["", "Predicted Real", "Predicted Fake"])
        writer.writerow(["Actual Real", tn, fp])
        writer.writerow(["Actual Fake", fn, tp])
    print(f"✓ CSV  → {csv_path}")
except Exception as e:
    print(f"  [ERROR] Failed to save CSV: {e}")

# Save confusion matrix PNG
png_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{DATASET_NAME}.png")
try:
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
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"✓ PNG  → {png_path}")
except Exception as e:
    print(f"  [ERROR] Failed to save PNG: {e}")

print("\nPipeline complete.")
