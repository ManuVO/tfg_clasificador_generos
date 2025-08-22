# src/eval.py
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from data.dataset import GenreDataset
from models.cnn_baseline import CNNBaseline

def load_items(csv_path):
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["label_idx"] = int(r["label_idx"])
            items.append(r)
    return items

def plot_confusion(cm, class_names, out_path):
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Matriz de confusión")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main(csv_path="data/processed/gtzan/metadata.csv",
         ckpt="experiments/checkpoints/best_cnn_gtzan.pt",
         num_classes=10):

    items = load_items(csv_path)
    test_items = [r for r in items if r["split"]=="test"]
    class_names = sorted(list({r["label"] for r in items}))

    ds = GenreDataset(test_items, "test", augment=False)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNBaseline(n_classes=num_classes).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            logits = model(xb)
            preds += logits.argmax(1).cpu().tolist()
            gts   += yb.tolist()

    print(classification_report(gts, preds, target_names=class_names, digits=4))
    cm = confusion_matrix(gts, preds)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, class_names, "reports/figures/confusion_matrix_gtzan.png")
    print("✅ Figura guardada en reports/figures/confusion_matrix_gtzan.png")

if __name__ == "__main__":
    main()
