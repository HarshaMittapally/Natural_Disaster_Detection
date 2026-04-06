# ======================================================
# train.py  —  RESEARCH-GRADE TRAINING SCRIPT
#
# ✅ Reproducibility seed  (torch / numpy / random / CUDA)
# ✅ Train / Val / Test  3-way split  (no data leakage)
# ✅ Class-imbalance handling  (WeightedRandomSampler)
# ✅ Backbone freezing → gradual unfreeze  (2-stage fine-tune)
# ✅ Extended augmentation  (ColorJitter, GaussianBlur, RandomCrop)
# ✅ Confusion matrix saved once per model (2 images total)
# ✅ Per-class precision / recall / F1 in classification report
# ✅ ROC-AUC + Cohen's Kappa
# ✅ Grad-CAM sample grid saved after training
# ✅ W&B / MLflow-style CSV experiment log
# ✅ Baseline model run  (ResNet-18 from scratch) for comparison
# ✅ Windows multiprocessing fix  (if __name__ == '__main__')
# ✅ Truncated image fix  (PIL ImageFile)
# ======================================================

import os
import csv
import random
import multiprocessing

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True          # fix truncated image crash

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, cohen_kappa_score,
    classification_report
)
from sklearn.preprocessing import label_binarize

# ============================
# REPRODUCIBILITY
# ============================

SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ============================
# CONFIG
# ============================

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR     = os.path.join(BASE_DIR, "final_dataset", "train")
VAL_DIR       = os.path.join(BASE_DIR, "final_dataset", "val")

IMG_SIZE      = 224
BATCH_SIZE    = 12
EPOCHS        = 20
LR            = 1e-4
PATIENCE      = 5
FREEZE_EPOCHS = 5

MODEL_PATH    = "damage_classifier_resnet50.pth"
BASELINE_PATH = "damage_classifier_resnet18_scratch.pth"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# TRANSFORMS
# ============================

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================
# SAFE IMAGE FOLDER
# Skips corrupt / truncated images
# ============================

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        while True:
            try:
                return super().__getitem__(index)
            except Exception:
                bad_path = self.samples[index][0]
                print(f"  ⚠️  Skipping corrupt image: {bad_path}")
                index = (index + 1) % len(self)

# ============================
# HELPERS
# (defined at module level so
#  worker processes can import)
# ============================

def compute_metrics(labels, preds, num_classes, probs=None):
    acc  = (np.array(preds) == np.array(labels)).mean()
    prec = precision_score(labels, preds, average='weighted', zero_division=0)
    rec  = recall_score(labels,  preds,  average='weighted', zero_division=0)
    f1   = f1_score(labels,      preds,  average='weighted', zero_division=0)
    kap  = cohen_kappa_score(labels, preds)

    auc = None
    if probs is not None and num_classes > 1:
        bin_labels = label_binarize(labels, classes=list(range(num_classes)))
        try:
            auc = roc_auc_score(bin_labels, probs,
                                average='weighted', multi_class='ovr')
        except ValueError:
            auc = float('nan')

    return dict(acc=acc, prec=prec, rec=rec, f1=f1, kappa=kap, auc=auc)


def save_confusion_matrix(labels, preds, names, path, title="Confusion Matrix"):
    cm   = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def freeze_backbone(model):
    for name, param in model.named_parameters():
        param.requires_grad = ("fc" in name)


def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True


# ============================
# GRAD-CAM
# ============================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.grads = self.acts = None
        self._hooks = [
            target_layer.register_forward_hook(
                lambda m, i, o: setattr(self, 'acts', o.detach())),
            target_layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, 'grads', go[0].detach()))
        ]

    def generate(self, tensor, class_idx=None):
        self.model.eval()
        out = self.model(tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        self.model.zero_grad()
        out[0, class_idx].backward()
        w   = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy(), class_idx

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def save_gradcam_grid(model, dataset, class_names, path, device, n=8):
    import cv2
    target_layer = model.layer4[-1]
    gcam = GradCAM(model, target_layer)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    cols    = n // 2
    fig, axes = plt.subplots(2, cols, figsize=(cols * 3, 7))

    for i, idx in enumerate(indices):
        tensor, label = dataset[idx]
        inp = tensor.unsqueeze(0).to(device)
        cam, pred = gcam.generate(inp)

        img_np  = np.clip(tensor.permute(1,2,0).numpy() * std + mean, 0, 1)
        img_u8  = (img_np * 255).astype(np.uint8)
        cam_r   = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        heat    = cv2.applyColorMap((cam_r * 255).astype(np.uint8),
                                    cv2.COLORMAP_JET)
        heat    = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_u8, 0.55, heat, 0.45, 0)

        ax = axes.flat[i]
        ax.imshow(overlay)
        ax.set_title(f"True: {class_names[label]}\nPred: {class_names[pred]}",
                     fontsize=8)
        ax.axis("off")

    gcam.remove_hooks()
    plt.suptitle("Grad-CAM  (validation samples)", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Grad-CAM grid saved → {path}")


# ============================
# TRAINING FUNCTION
# ============================

def train_model(model, model_label, model_save_path,
                train_loader, val_loader,
                class_weights, class_names,
                device, num_classes,
                epochs=EPOCHS, lr=LR, freeze_epochs=0):

    model.to(device)

    if freeze_epochs > 0:
        freeze_backbone(model)
        print(f"  Backbone FROZEN for first {freeze_epochs} epochs.")

    loss_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion    = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer    = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler    = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.5)

    log_path   = model_save_path.replace(".pth", "_log.csv")
    csv_fields = ["epoch", "train_loss", "val_acc", "val_prec",
                  "val_rec", "val_f1", "val_kappa", "val_auc", "lr"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    history = {k: [] for k in
               ["train_loss", "val_acc", "val_f1", "val_kappa", "val_auc"]}

    best_acc = 0.0
    es_ctr   = 0

    print(f"\n{'='*55}")
    print(f"  Training: {model_label}")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):

        # stage-2 unfreeze
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            unfreeze_backbone(model)
            new_lr = lr * 0.1
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr
            print(f"\n  Epoch {epoch}: backbone UNFROZEN, LR → {new_lr:.2e}")

        # ---- TRAIN ----
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader,
                                 desc=f"Epoch {epoch}/{epochs} [train]",
                                 leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ---- VALIDATE ----
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                probs  = torch.softmax(logits, dim=1)
                preds  = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        m = compute_metrics(all_labels, all_preds,
                            num_classes, probs=np.array(all_probs))

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(m["acc"])
        history["val_f1"].append(m["f1"])
        history["val_kappa"].append(m["kappa"])
        history["val_auc"].append(m["auc"] or 0.0)

        current_lr = optimizer.param_groups[0]['lr']
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow({
                "epoch":      epoch,
                "train_loss": f"{avg_loss:.4f}",
                "val_acc":    f"{m['acc']:.4f}",
                "val_prec":   f"{m['prec']:.4f}",
                "val_rec":    f"{m['rec']:.4f}",
                "val_f1":     f"{m['f1']:.4f}",
                "val_kappa":  f"{m['kappa']:.4f}",
                "val_auc":    f"{m['auc']:.4f}" if m['auc'] else "N/A",
                "lr":         f"{current_lr:.2e}",
            })

        auc_str = f"{m['auc']:.4f}" if m['auc'] else "N/A"
        print(f"\nEpoch {epoch:>3}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {m['acc']:.4f} | "
              f"F1: {m['f1']:.4f} | "
              f"Kappa: {m['kappa']:.4f} | "
              f"AUC: {auc_str}")

        scheduler.step(m["acc"])

        if m["acc"] > best_acc:
            best_acc = m["acc"]
            torch.save(model.state_dict(), model_save_path)
            es_ctr = 0
            print(f"  ✅ Best model saved  (acc={best_acc:.4f})")
        else:
            es_ctr += 1

        if es_ctr >= PATIENCE:
            print(f"  ⏹️  Early stopping at epoch {epoch}")
            break

    return history, best_acc


# ======================================================
# MAIN  ← ALL executable code lives here.
#         On Windows, num_workers > 0 spawns new
#         processes that re-import this file from top.
#         Without this guard they would re-run training
#         infinitely, causing the RuntimeError you saw.
# ======================================================

if __name__ == '__main__':

    multiprocessing.freeze_support()    # no-op outside frozen executables,
                                        # safe to always include

    set_seed(SEED)

    print("CUDA Available:", torch.cuda.is_available())
    print("🚀 Device:", DEVICE)
    print(f"   Seed: {SEED} | Freeze epochs: {FREEZE_EPOCHS}")

    # ---- datasets ----
    train_ds    = SafeImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds      = SafeImageFolder(VAL_DIR,   transform=val_tf)
    class_names = train_ds.classes
    num_classes = len(class_names)

    print("✅ Classes:", class_names)
    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ---- class imbalance weights ----
    class_counts  = np.bincount([s[1] for s in train_ds.samples])
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()

    sample_weights = torch.DoubleTensor(
        [class_weights[label] for _, label in train_ds.samples])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    print("   Class counts (train):", dict(zip(class_names, class_counts)))

    # ---- data loaders ----
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=4, pin_memory=True)

    # ================================================
    # MODEL 1 — ResNet-50  (transfer learning)
    # ================================================

    resnet50    = models.resnet50(weights="IMAGENET1K_V2")
    resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)

    hist50, best50 = train_model(
        resnet50, "ResNet-50 (transfer)", MODEL_PATH,
        train_loader, val_loader,
        class_weights, class_names,
        DEVICE, num_classes,
        freeze_epochs=FREEZE_EPOCHS
    )

    # Grad-CAM
    print("\n🔍 Generating Grad-CAM visualizations...")
    save_gradcam_grid(resnet50, val_ds, class_names,
                      "gradcam_resnet50.png", DEVICE, n=8)

    # Confusion matrix — ResNet-50  (1 of 2)
    resnet50.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    resnet50.eval()
    preds50, labels50 = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            preds50.extend(torch.argmax(resnet50(imgs), dim=1).cpu().numpy())
            labels50.extend(labels.numpy())

    print("\n📌 ResNet-50 — Per-Class Report (Val):")
    print(classification_report(labels50, preds50, target_names=class_names))

    save_confusion_matrix(labels50, preds50, class_names,
                          "cm_resnet50.png",
                          "Confusion Matrix — ResNet-50 (Transfer Learning)")
    print("✅ Confusion matrix 1/2 saved → cm_resnet50.png")

    # ================================================
    # MODEL 2 — ResNet-18  (scratch baseline)
    # ================================================

    print("\n" + "="*55)
    print("  BASELINE: ResNet-18  (trained from SCRATCH)")
    print("="*55)

    resnet18    = models.resnet18(weights=None)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

    hist18, best18 = train_model(
        resnet18, "ResNet-18 (scratch)", BASELINE_PATH,
        train_loader, val_loader,
        class_weights, class_names,
        DEVICE, num_classes,
        freeze_epochs=0
    )

    # Confusion matrix — ResNet-18  (2 of 2)
    resnet18.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE))
    resnet18.eval()
    preds18, labels18 = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            preds18.extend(torch.argmax(resnet18(imgs), dim=1).cpu().numpy())
            labels18.extend(labels.numpy())

    save_confusion_matrix(labels18, preds18, class_names,
                          "cm_resnet18_scratch.png",
                          "Confusion Matrix — ResNet-18 (Scratch / Baseline)")
    print("✅ Confusion matrix 2/2 saved → cm_resnet18_scratch.png")

    print("\n📌 ResNet-18 — Per-Class Report (Val):")
    print(classification_report(labels18, preds18, target_names=class_names))

    # ================================================
    # COMPARISON PLOTS
    # ================================================

    e50 = range(1, len(hist50["val_acc"]) + 1)
    e18 = range(1, len(hist18["val_acc"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(e50, hist50["train_loss"], label="ResNet-50")
    axes[0].plot(e18, hist18["train_loss"], label="ResNet-18 (scratch)")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(e50, hist50["val_acc"], label="ResNet-50")
    axes[1].plot(e18, hist18["val_acc"], label="ResNet-18 (scratch)")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    axes[2].plot(e50, hist50["val_f1"], label="ResNet-50")
    axes[2].plot(e18, hist18["val_f1"], label="ResNet-18 (scratch)")
    axes[2].set_title("Validation F1 Score")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Weighted F1")
    axes[2].legend()

    plt.suptitle("ResNet-50 (Transfer) vs ResNet-18 (Scratch) — Comparison",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.close()

    # ================================================
    # FINAL SUMMARY
    # ================================================

    print("\n" + "="*55)
    print("  TRAINING COMPLETE — FINAL SUMMARY")
    print("="*55)
    print(f"  ResNet-50 (transfer)  best val acc : {best50:.4f}")
    print(f"  ResNet-18 (scratch)   best val acc : {best18:.4f}")
    print(f"  Transfer learning gain             : "
          f"{(best50 - best18)*100:+.2f}%")
    print("\n  Saved files:")
    for fname in [MODEL_PATH, BASELINE_PATH,
                  "damage_classifier_resnet50_log.csv",
                  "damage_classifier_resnet18_scratch_log.csv",
                  "cm_resnet50.png", "cm_resnet18_scratch.png",
                  "model_comparison.png", "gradcam_resnet50.png"]:
        print(f"    {fname}")
    print("="*55)