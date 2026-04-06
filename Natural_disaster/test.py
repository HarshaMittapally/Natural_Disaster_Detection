# ======================================================
# test.py  —  RESEARCH-GRADE EVALUATION SCRIPT
#
# Additions over previous version:
# ✅ Reproducibility seed
# ✅ Loads UNSEEN test split  (final_dataset/test/)
# ✅ Accuracy / Precision / Recall / F1  (weighted + per-class)
# ✅ Cohen's Kappa
# ✅ ROC-AUC  (one-vs-rest, multi-class)
# ✅ Confusion matrix  (saved as PNG)
# ✅ Grad-CAM overlay grid  (saved as PNG)
# ✅ PDF report  (all metrics + confusion matrix image)
# ✅ GUI predictor with damage overlay
# ======================================================

import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")          # keeps PDF/PNG generation headless
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # ← fixes OSError on truncated files

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    cohen_kappa_score,
)
from sklearn.preprocessing import label_binarize

from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

import cv2

# ============================
# REPRODUCIBILITY
# ============================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ============================
# CONFIG
# ============================

TEST_DIR   = r"./final_dataset/test"      # unseen test split
MODEL_PATH = "damage_classifier_resnet50.pth"

PDF_PATH  = "Damage_Test_Report.pdf"
CM_PATH   = "Confusion_Matrix_Test.png"
GCAM_PATH = "GradCAM_Test.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("\n🚀 Running Evaluation on UNSEEN Test Set")
print("   Device:", DEVICE)

# ============================
# TRANSFORM
# ============================

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================
# LOAD TEST DATASET
# ============================

class SafeImageFolder(datasets.ImageFolder):
    """Silently skips corrupt or unreadable images."""
    def __getitem__(self, index):
        while True:
            try:
                return super().__getitem__(index)
            except Exception:
                bad_path = self.samples[index][0]
                print(f"  ⚠️  Skipping corrupt image: {bad_path}")
                index = (index + 1) % len(self)

test_ds     = SafeImageFolder(TEST_DIR, transform=tf)
class_names = test_ds.classes
num_classes = len(class_names)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

print("✅ Classes:", class_names)
print("   Total test images:", len(test_ds))

# ============================
# LOAD MODEL
# ============================

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

print("✅ Model loaded:", MODEL_PATH)

# ============================
# INFERENCE
# ============================

all_preds, all_labels, all_probs = [], [], []

print("\n📌 Running inference on test set...")
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs   = imgs.to(DEVICE)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)
        preds  = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_probs  = np.array(all_probs)
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ============================
# METRICS
# ============================

acc   = accuracy_score(all_labels, all_preds)
kappa = cohen_kappa_score(all_labels, all_preds)              # ← NEW

bin_labels = label_binarize(all_labels, classes=list(range(num_classes)))
try:
    auc = roc_auc_score(bin_labels, all_probs,               # ← NEW
                        average='weighted', multi_class='ovr')
except ValueError:
    auc = float('nan')

report = classification_report(
    all_labels, all_preds,
    target_names=class_names,
    digits=4
)

print("\n" + "="*55)
print("  TEST SET RESULTS")
print("="*55)
print(f"  Accuracy     : {acc:.4f}")
print(f"  Cohen's Kappa: {kappa:.4f}")
print(f"  ROC-AUC      : {auc:.4f}" if not np.isnan(auc) else "  ROC-AUC: N/A")
print("\n📌 Per-Class Classification Report:")
print(report)

# ============================
# CONFUSION MATRIX             ← enhanced
# ============================

cm   = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap="Blues", ax=ax, colorbar=False)
ax.set_title("Confusion Matrix — Unseen Test Set")
plt.tight_layout()
plt.savefig(CM_PATH, dpi=150)
plt.close()
print(f"\n✅ Confusion matrix saved → {CM_PATH}")

# ============================
# GRAD-CAM ON TEST SAMPLES     ← NEW
# ============================

class GradCAM:
    def __init__(self, mdl, layer):
        self.grads = self.acts = None
        self._h = [
            layer.register_forward_hook(
                lambda m, i, o: setattr(self, 'acts', o.detach())),
            layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, 'grads', go[0].detach()))
        ]

    def generate(self, tensor, cls=None):
        model.eval()
        out = model(tensor)
        if cls is None:
            cls = out.argmax(dim=1).item()
        model.zero_grad()
        out[0, cls].backward()
        w   = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(dim=1)).squeeze()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam.cpu().numpy(), cls

    def remove(self):
        for h in self._h:
            h.remove()


def save_gradcam_grid(dataset, path, n=8):
    gcam = GradCAM(model, model.layer4[-1])
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))
    cols = n // 2
    fig, axes = plt.subplots(2, cols, figsize=(cols * 3, 7))

    for i, idx in enumerate(idxs):
        tensor, label = dataset[idx]
        inp = tensor.unsqueeze(0).to(DEVICE)
        cam, pred = gcam.generate(inp)

        img_np = np.clip(tensor.permute(1,2,0).numpy() * std + mean, 0, 1)
        img_u8 = (img_np * 255).astype(np.uint8)

        cam_r = cv2.resize(cam, (224, 224))
        heat  = cv2.applyColorMap((cam_r * 255).astype(np.uint8),
                                  cv2.COLORMAP_JET)
        heat  = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        over  = cv2.addWeighted(img_u8, 0.55, heat, 0.45, 0)

        ax = axes.flat[i]
        ax.imshow(over)
        correct = "✓" if pred == label else "✗"
        ax.set_title(
            f"{correct} True: {class_names[label]}\n"
            f"   Pred: {class_names[pred]}", fontsize=8)
        ax.axis("off")

    gcam.remove()
    plt.suptitle("Grad-CAM  —  Test Set Samples", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Grad-CAM grid saved → {path}")


save_gradcam_grid(test_ds, GCAM_PATH, n=8)

# ============================
# PDF REPORT                   ← enhanced
# ============================

def generate_pdf():
    c = rl_canvas.Canvas(PDF_PATH, pagesize=letter)
    W, H = letter

    def header(title, y=750):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, title)
        c.setFont("Helvetica", 11)
        return y - 30

    # --- Page 1: Metrics ---
    y = header("Earthquake Damage Classification — Test Report")
    c.drawString(50, y, f"Model: ResNet-50 (Transfer Learning)")
    y -= 20
    c.drawString(50, y, f"Test Set : {TEST_DIR}    |    Images: {len(test_ds)}")
    y -= 30

    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Overall Metrics")
    y -= 20
    c.setFont("Helvetica", 11)

    for label, val in [
        ("Accuracy", f"{acc:.4f}"),
        ("Cohen's Kappa", f"{kappa:.4f}"),
        ("ROC-AUC (weighted OvR)",
         f"{auc:.4f}" if not np.isnan(auc) else "N/A"),
    ]:
        c.drawString(60, y, f"{label:<30}{val}")
        y -= 18

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Per-Class Classification Report")
    y -= 20
    c.setFont("Courier", 9)

    for line in report.split("\n"):
        c.drawString(50, y, line)
        y -= 13
        if y < 80:
            c.showPage()
            y = 750
            c.setFont("Courier", 9)

    c.showPage()

    # --- Page 2: Confusion Matrix ---
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 750, "Confusion Matrix")
    if os.path.exists(CM_PATH):
        c.drawImage(ImageReader(CM_PATH), 50, 380,
                    width=500, height=340, preserveAspectRatio=True)

    # --- Page 3: Grad-CAM ---
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 750, "Grad-CAM Visualizations")
    c.setFont("Helvetica", 10)
    c.drawString(50, 730,
                 "Highlighted regions show where the model focuses to make predictions.")
    if os.path.exists(GCAM_PATH):
        c.drawImage(ImageReader(GCAM_PATH), 30, 200,
                    width=540, height=500, preserveAspectRatio=True)

    c.save()
    print(f"📄 PDF report saved → {PDF_PATH}")


generate_pdf()

# ============================
# GUI PREDICTOR  (unchanged logic,
# but now shows confidence %)  ← enhanced
# ============================

def run_gui():
    import tkinter as tk
    from tkinter import filedialog

    matplotlib.use("TkAgg")    # switch backend for GUI display
    import matplotlib.pyplot as plt_gui

    LABEL_COLORS = {
        "Collapsed": (0, 0, 255),     # red in BGR
        "Damaged"  : (0, 165, 255),   # orange
        "Intact"   : (255, 0, 0),     # blue
    }

    def overlay_damage(img_cv, label):
        color   = LABEL_COLORS.get(label, (200, 200, 200))
        overlay = img_cv.copy()
        cv2.rectangle(overlay, (0, 0), (224, 224), color, -1)
        return cv2.addWeighted(img_cv, 0.7, overlay, 0.3, 0)

    def upload_predict():
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return

        img     = Image.open(path).convert("RGB")
        tensor  = tf(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred   = probs.argmax()

        label      = class_names[pred]
        confidence = probs[pred] * 100

        # overlay
        img_cv  = cv2.resize(cv2.imread(path), (224, 224))
        mapped  = overlay_damage(img_cv, label)

        fig, axes = plt_gui.subplots(1, 2, figsize=(9, 4))
        axes[0].imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB))
        axes[1].set_title(
            f"Prediction: {label}  ({confidence:.1f}%)")
        axes[1].axis("off")

        plt_gui.tight_layout()
        plt_gui.show()

        result_var.set(f"Prediction: {label}  ({confidence:.1f}% confidence)")

        # per-class confidence bar
        bar_win = tk.Toplevel(app)
        bar_win.title("Class Probabilities")
        for i, (cname, prob) in enumerate(zip(class_names, probs)):
            tk.Label(bar_win, text=f"{cname}:", anchor="w",
                     width=15).grid(row=i, column=0, padx=8, pady=4)
            tk.Label(bar_win,
                     text=f"{prob*100:.1f}%",
                     width=8).grid(row=i, column=1)
            bar = tk.Canvas(bar_win, width=200, height=18,
                            bg="white", bd=1, relief="solid")
            bar.grid(row=i, column=2, padx=8)
            bar.create_rectangle(0, 0, int(prob * 200), 18,
                                 fill="#2196F3", outline="")

    app = tk.Tk()
    app.title("Earthquake Damage Prediction System")
    app.geometry("480x160")

    tk.Button(
        app, text="Upload Image & Predict",
        command=upload_predict,
        font=("Arial", 14),
        padx=12, pady=6
    ).pack(pady=20)

    result_var = tk.StringVar(value="Prediction: —")
    tk.Label(app, textvariable=result_var,
             font=("Arial", 13)).pack(pady=6)

    app.mainloop()


if __name__ == "__main__":
    run_gui()