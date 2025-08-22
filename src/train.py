# src/train.py
import csv
from pathlib import Path

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

from data.dataset import GenreDataset
from models.cnn_baseline import CNNBaseline

def load_items_from_csv(csv_path: str):
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label_idx"] = int(row["label_idx"])
            items.append(row)
    return items

def main(csv_path="data/processed/gtzan/metadata.csv",
         epochs=40, batch_size=64, lr=1e-3, num_classes=10, device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    items = load_items_from_csv(csv_path)
    train_items = [r for r in items if r["split"]=="train"]
    val_items   = [r for r in items if r["split"]=="val"]

    train_ds = GenreDataset(train_items, "train", augment=True)
    val_ds   = GenreDataset(val_items, "val", augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = CNNBaseline(n_classes=num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    writer = SummaryWriter("experiments/logs/gtzan_cnn")

    best_acc = 0.0
    Path("experiments/checkpoints").mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward(); opt.step()

        # validación
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                logits = model(xb)
                preds += logits.argmax(1).cpu().tolist()
                gts   += yb.tolist()
        acc = accuracy_score(gts, preds)
        writer.add_scalar("val/accuracy", acc, epoch)
        print(f"[{epoch+1:03d}] val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "experiments/checkpoints/best_cnn_gtzan.pt")

    writer.close()
    print(f"✅ Entrenamiento finalizado. Mejor val_acc={best_acc:.4f}")

if __name__ == "__main__":
    main()
