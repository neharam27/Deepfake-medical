"""
script_3_rsna_pneumonia.py
Leakage-free deepfake detection pipeline for RSNA Pneumonia Detection Challenge.

Dataset: RSNA Pneumonia Detection Challenge (Kaggle)
  URL: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge
  Images are stored as DICOM (.dcm) files.

Expected layout:
  DATA_PATH/
    stage_2_train_images/    (or stage_2_test_images/ – DCM files)
    stage_2_train_labels.csv (optional, used only to gather image paths)

Requires:  pip install pydicom

Before running, set DATA_PATH to the root of your RSNA download.
"""

# ============================================================
# IMPORTS
# ============================================================
import os, sys, csv, random
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

try:
    import pydicom
except ImportError:
    sys.exit("[ERROR] pydicom is required.  Run:  pip install pydicom")

# ============================================================
# USER CONFIGURATION
# ============================================================
DATA_PATH  = os.environ.get("RSNA_PATH", "./rsna_pneumonia_data")
OUTPUT_DIR = "."
WORK_DIR   = "./work_rsna_pneumonia"

# ============================================================
# PIPELINE CONSTANTS
# ============================================================
DATASET_NAME  = "rsna_pneumonia"
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

random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE); torch.manual_seed(RANDOM_STATE)

DATASET_DIR = os.path.join(WORK_DIR, "datasets")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
VAL_DIR     = os.path.join(DATASET_DIR, "val")
TEST_DIR    = os.path.join(DATASET_DIR, "test")
FAKES_DIR   = os.path.join(WORK_DIR, "generated_fakes")
FAKES_VAL   = os.path.join(FAKES_DIR, "val")
FAKES_TEST  = os.path.join(FAKES_DIR, "test")
MODELS_DIR  = os.path.join(WORK_DIR, "models")
CKPT_DIR    = os.path.join(WORK_DIR, "checkpoints")

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR, FAKES_VAL, FAKES_TEST, MODELS_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# STEP 0 – Load DICOM images, convert to PNG, split 60/20/20
# ============================================================
print("=" * 70)
print("STEP 0: DATA LOADING & SPLITTING (RSNA Pneumonia)")
print("=" * 70)

def collect_dcm_paths(data_path):
    """Recursively find all .dcm files under data_path."""
    paths = []
    for root, _dirs, files in os.walk(data_path):
        for fname in files:
            if fname.lower().endswith(".dcm"):
                paths.append(os.path.join(root, fname))
    return sorted(paths)

def dicom_to_gray(dcm_path):
    """Read a DICOM file and return a uint8 grayscale numpy array, or None on failure."""
    try:
        ds  = pydicom.dcmread(dcm_path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        # Normalise to 0-255
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)
        arr = arr.astype(np.uint8)
        # Handle multi-channel DICOM (RGB / YBR)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return arr
    except Exception:
        return None

all_dcm = collect_dcm_paths(DATA_PATH)
if len(all_dcm) == 0:
    sys.exit(
        f"[ERROR] No .dcm files found under DATA_PATH='{DATA_PATH}'.\n"
        "Set DATA_PATH to the RSNA Pneumonia dataset root."
    )

print(f"Found {len(all_dcm)} DICOM files")

indices = np.arange(len(all_dcm))
train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=RANDOM_STATE)
val_idx,   test_idx = train_test_split(temp_idx, test_size=0.5, random_state=RANDOM_STATE)
print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

def dcm_copy_resize(dcm_paths, dest_dir, img_size, desc):
    os.makedirs(dest_dir, exist_ok=True)
    saved = 0
    for i, p in enumerate(tqdm(dcm_paths, desc=desc)):
        img = dicom_to_gray(p)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        base = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(dest_dir, f"{i:06d}_{base}.png"), img)
        saved += 1
    return saved

dcm_copy_resize([all_dcm[i] for i in train_idx], TRAIN_DIR, IMG_SIZE_GAN, "Copy train")
dcm_copy_resize([all_dcm[i] for i in val_idx],   VAL_DIR,   IMG_SIZE_GAN, "Copy val")
dcm_copy_resize([all_dcm[i] for i in test_idx],  TEST_DIR,  IMG_SIZE_GAN, "Copy test")

train_files = set(os.listdir(TRAIN_DIR)); val_files = set(os.listdir(VAL_DIR)); test_files = set(os.listdir(TEST_DIR))
assert not (train_files & val_files);  assert not (train_files & test_files); assert not (val_files & test_files)
print("✓ No overlap verified")

# ============================================================
# GAN ARCHITECTURE
# ============================================================
class ImprovedGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_size=64):
        super().__init__()
        self.fc    = nn.Linear(latent_dim, 512 * 4 * 4)
        self.bn_fc = nn.BatchNorm1d(512 * 4 * 4)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1,bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,4,2,1,bias=False), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d( 64, 32,4,2,1,bias=False), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32,1,3,1,1), nn.Tanh(),
        )
    def forward(self, z):
        x = self.fc(z); x = self.bn_fc(x); x = x.view(x.size(0),512,4,4); return self.model(x)


class ImprovedDiscriminator(nn.Module):
    def __init__(self, img_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32,4,2,1),                              nn.LeakyReLU(0.2,True),
            nn.Conv2d(32, 64,4,2,1,bias=False), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2,True),
            nn.Conv2d(64,128,4,2,1,bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2,True),
            nn.Conv2d(128,256,4,2,1,bias=False),nn.BatchNorm2d(256), nn.LeakyReLU(0.2,True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256,1)
    def forward(self, x):
        x = self.model(x); x = x.view(x.size(0),-1); return self.fc(x)


class MedicalImageDataset(Dataset):
    def __init__(self, img_dir, img_size=64):
        self.img_dir = img_dir; self.img_size = img_size
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])
    def __len__(self): return len(self.img_files)
    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.img_files[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None: return torch.zeros(1,self.img_size,self.img_size)
        img = img.astype(np.float32)/255.0; img = (img-0.5)/0.5
        return torch.from_numpy(img).unsqueeze(0)

# ============================================================
# STEP 1 – Train GAN
# ============================================================
print("=" * 70)
print("STEP 1: GAN TRAINING")
print("=" * 70)

generator     = ImprovedGenerator(LATENT_DIM,IMG_SIZE_GAN).to(DEVICE)
discriminator = ImprovedDiscriminator(IMG_SIZE_GAN).to(DEVICE)
criterion     = nn.BCEWithLogitsLoss()
opt_g  = optim.Adam(generator.parameters(),     lr=LR_GAN,betas=(0.5,0.999))
opt_d  = optim.Adam(discriminator.parameters(), lr=LR_GAN,betas=(0.5,0.999))
sched_g = optim.lr_scheduler.CosineAnnealingLR(opt_g,T_max=GAN_EPOCHS)
sched_d = optim.lr_scheduler.CosineAnnealingLR(opt_d,T_max=GAN_EPOCHS)

train_ds     = MedicalImageDataset(TRAIN_DIR, IMG_SIZE_GAN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
print(f"GAN training on {len(train_ds)} images | device={DEVICE}")

for epoch in range(GAN_EPOCHS):
    for real_imgs in tqdm(train_loader, desc=f"GAN {epoch+1}/{GAN_EPOCHS}", leave=False):
        real_imgs = real_imgs.to(DEVICE); bs = real_imgs.size(0)
        opt_d.zero_grad()
        ld = criterion(discriminator(real_imgs), torch.ones(bs,1,device=DEVICE)) + \
             criterion(discriminator(generator(torch.randn(bs,LATENT_DIM,device=DEVICE)).detach()),
                       torch.zeros(bs,1,device=DEVICE))
        ld.backward(); opt_d.step()
        opt_g.zero_grad()
        lg = criterion(discriminator(generator(torch.randn(bs,LATENT_DIM,device=DEVICE))),
                       torch.ones(bs,1,device=DEVICE))
        lg.backward(); opt_g.step()
    sched_g.step(); sched_d.step()

torch.save(generator.state_dict(), os.path.join(CKPT_DIR,"generator_train_only.pt"))
print("✓ Generator saved")

# ============================================================
# STEP 2 – Generate fakes
# ============================================================
print("=" * 70)
print("STEP 2: GENERATE FAKE IMAGES")
print("=" * 70)

generator.eval()

def generate_fakes(dest_dir, count, prefix):
    os.makedirs(dest_dir, exist_ok=True)
    with torch.no_grad():
        for i in tqdm(range(0,count,BATCH_SIZE), desc=f"Fakes {prefix}"):
            bs   = min(BATCH_SIZE,count-i)
            imgs = generator(torch.randn(bs,LATENT_DIM,device=DEVICE))
            for j in range(bs):
                img = ((imgs[j].cpu().numpy().squeeze()+1)/2*255).astype(np.uint8)
                cv2.imwrite(os.path.join(dest_dir,f"fake_{prefix}_{i+j:05d}.png"), img)

val_count = len(os.listdir(VAL_DIR)); test_count = len(os.listdir(TEST_DIR))
generate_fakes(FAKES_VAL, val_count, "val"); generate_fakes(FAKES_TEST, test_count, "test")
print(f"✓ {val_count} val fakes | {test_count} test fakes")

# ============================================================
# DETECTOR
# ============================================================
class RealFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, img_size=256):
        self.img_size = img_size; self.images=[]; self.labels=[]
        for f in sorted(os.listdir(real_dir)):
            if f.lower().endswith(".png"): self.images.append(os.path.join(real_dir,f)); self.labels.append(0)
        for f in sorted(os.listdir(fake_dir)):
            if f.lower().endswith(".png"): self.images.append(os.path.join(fake_dir,f)); self.labels.append(1)
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        if img is None: return torch.zeros(1,self.img_size,self.img_size), self.labels[idx]
        img = cv2.resize(img,(self.img_size,self.img_size)).astype(np.float32)/255.0
        return torch.from_numpy(img).unsqueeze(0), self.labels[idx]


class MedicalDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None); resnet.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(True),nn.Dropout(0.5),
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(True),nn.Dropout(0.3),
            nn.Linear(128,2),
        )
    def forward(self,x): return self.classifier(torch.flatten(self.backbone(x),1))

# ============================================================
# STEP 3 – Train detector
# ============================================================
print("=" * 70)
print("STEP 3: DETECTOR TRAINING")
print("=" * 70)

val_ds = RealFakeDataset(VAL_DIR, FAKES_VAL, IMG_SIZE_DET)
idx_all= np.arange(len(val_ds)); lbl_arr=np.array(val_ds.labels)
tr_idx,iv_idx = train_test_split(idx_all, test_size=0.2, random_state=RANDOM_STATE, stratify=lbl_arr)
tr_loader = DataLoader(torch.utils.data.Subset(val_ds,tr_idx), batch_size=BATCH_SIZE, shuffle=True)
iv_loader = DataLoader(torch.utils.data.Subset(val_ds,iv_idx), batch_size=BATCH_SIZE)

detector   = MedicalDeepfakeDetector().to(DEVICE)
crit_det   = nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.2]).to(DEVICE))
opt_det    = optim.Adam(detector.parameters(), lr=LR_DET, weight_decay=1e-5)
sched_det  = optim.lr_scheduler.ReduceLROnPlateau(opt_det, mode="max", factor=0.5, patience=3)
best_auc   = 0.0; patience_cnt = 0

for epoch in range(DET_EPOCHS):
    detector.train()
    for imgs,lbls in tqdm(tr_loader, desc=f"Det {epoch+1}/{DET_EPOCHS}", leave=False):
        imgs,lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        loss = crit_det(detector(imgs), lbls)
        opt_det.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(detector.parameters(),1.0); opt_det.step()

    detector.eval(); iv_preds,iv_labels,iv_probs=[],[],[]
    with torch.no_grad():
        for imgs,lbls in iv_loader:
            logits=detector(imgs.to(DEVICE)); probs=torch.softmax(logits,1)
            iv_preds.extend(torch.argmax(logits,1).cpu().numpy())
            iv_labels.extend(lbls.numpy()); iv_probs.extend(probs[:,1].cpu().numpy())

    vauc=roc_auc_score(iv_labels,iv_probs); sched_det.step(vauc)
    if vauc>best_auc:
        best_auc=vauc; patience_cnt=0
        torch.save(detector.state_dict(), os.path.join(MODELS_DIR,"best_detector.pt"))
    else:
        patience_cnt+=1
        if patience_cnt>=5: print(f"Early stopping at epoch {epoch+1}"); break

print(f"✓ Best val AUC: {best_auc:.4f}")

# ============================================================
# STEP 4 – Evaluate
# ============================================================
print("=" * 70)
print("STEP 4: FINAL EVALUATION ON TEST SET")
print("=" * 70)

detector.load_state_dict(torch.load(os.path.join(MODELS_DIR,"best_detector.pt"), map_location=DEVICE))
detector.eval()

test_images=[]; test_labels=[]
for f in sorted(os.listdir(TEST_DIR)):
    if not f.lower().endswith(".png"): continue
    img=cv2.imread(os.path.join(TEST_DIR,f),cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    img=cv2.resize(img,(IMG_SIZE_DET,IMG_SIZE_DET)).astype(np.float32)/255.0
    test_images.append(torch.from_numpy(img).unsqueeze(0)); test_labels.append(0)
for f in sorted(os.listdir(FAKES_TEST)):
    if not f.lower().endswith(".png"): continue
    img=cv2.imread(os.path.join(FAKES_TEST,f),cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    img=cv2.resize(img,(IMG_SIZE_DET,IMG_SIZE_DET)).astype(np.float32)/255.0
    test_images.append(torch.from_numpy(img).unsqueeze(0)); test_labels.append(1)

all_preds=[]; all_probs=[]
with torch.no_grad():
    for i in tqdm(range(0,len(test_images),BATCH_SIZE), desc="Inference"):
        batch=torch.stack(test_images[i:i+BATCH_SIZE]).to(DEVICE)
        logits=detector(batch); probs=torch.softmax(logits,1)
        all_preds.extend(torch.argmax(logits,1).cpu().numpy()); all_probs.extend(probs[:,1].cpu().numpy())

all_labels=np.array(test_labels); all_preds=np.array(all_preds); all_probs=np.array(all_probs)
accuracy=accuracy_score(all_labels,all_preds)
precision=precision_score(all_labels,all_preds,zero_division=0)
recall=recall_score(all_labels,all_preds,zero_division=0)
f1=f1_score(all_labels,all_preds,zero_division=0)
auc=roc_auc_score(all_labels,all_probs)
cm=confusion_matrix(all_labels,all_preds); tn,fp,fn,tp=cm.ravel()

print(classification_report(all_labels,all_preds,target_names=["Real","Fake"]))
print(f"Accuracy={accuracy:.4f}  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")

csv_path=os.path.join(OUTPUT_DIR,f"confusion_matrix_{DATASET_NAME}.csv")
with open(csv_path,"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["","Predicted Real","Predicted Fake"])
    w.writerow(["Actual Real",tn,fp]); w.writerow(["Actual Fake",fn,tp])
print(f"✓ CSV → {csv_path}")

fig,ax=plt.subplots(figsize=(6,5))
im=ax.imshow(cm,cmap=plt.cm.Blues); plt.colorbar(im,ax=ax)
ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Real","Fake"],yticklabels=["Real","Fake"],
       ylabel="True label",xlabel="Predicted label",title=f"Confusion Matrix – {DATASET_NAME}")
thresh=cm.max()/2.0
for r in range(2):
    for c in range(2):
        ax.text(c,r,format(cm[r,c],"d"),ha="center",va="center",
                color="white" if cm[r,c]>thresh else "black")
plt.tight_layout()
png_path=os.path.join(OUTPUT_DIR,f"confusion_matrix_{DATASET_NAME}.png")
plt.savefig(png_path,dpi=150); plt.close()
print(f"✓ PNG → {png_path}\nPipeline complete.")
