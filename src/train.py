# # src/train.py

import csv
import random
import numpy as np
import shutil
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

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


def main(config_path="configs/gtzan_cnn.yaml"):

    # Cargar configuración YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Semilla para reproducibilidad
    seed = config["training"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Leer hiperparámetros del YAML
    csv_path = config["dataset"]["csv_path"]
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]
    num_classes = config["model"]["num_classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cargar datos
    items = load_items_from_csv(csv_path)
    train_items = [r for r in items if r["split"] == "train"]
    val_items = [r for r in items if r["split"] == "val"]
    train_ds = GenreDataset(train_items, "train", augment=True)
    val_ds = GenreDataset(val_items, "val", augment=False)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )
    model = CNNBaseline(n_classes=num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    # Preparar carpeta de experimento con timestamp
    experiment_name = Path(config_path).stem  # e.g., "gtzan_cnn"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = Path("experiments") / experiment_name / f"run_{timestamp}"
    logs_dir = run_dir / "logs"
    ckpt_dir = run_dir / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Guardar copia de la configuración usada
    shutil.copy(config_path, run_dir / "config.yaml")

    # Inicializar TensorBoard logger en la carpeta de logs de este run
    writer = SummaryWriter(str(logs_dir))

    best_acc = 0.0
    best_epoch = 0
    # Preparar CSV para métricas
    metrics_file = open(run_dir / "metrics.csv", "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(
        [
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "val_f1_macro",
        ]
    )
    for epoch in range(epochs):
        model.train()
        # Inicializar acumuladores
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            # Acumular métricas de entrenamiento
            batch_size_n = yb.size(0)
            total_loss += loss.item() * batch_size_n
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_samples += batch_size_n
        # Calcular métricas de entrenamiento de la época
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # Evaluación en validación
        model.eval()
        val_preds, val_gts = [], []
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                # Acumular loss de validación
                val_loss_sum += crit(logits, yb).item() * yb.size(0)
                # Recolectar predicciones para accuracy/F1
                val_preds += logits.argmax(1).cpu().tolist()
                val_gts += yb.cpu().tolist()
        val_loss = val_loss_sum / len(val_gts)
        val_acc = accuracy_score(val_gts, val_preds)
        val_f1 = f1_score(val_gts, val_preds, average="macro")

        # Registrar en TensorBoard
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("val/F1_macro", val_f1, epoch)
        # Registrar en CSV
        metrics_writer.writerow(
            [
                epoch + 1,
                f"{train_loss:.4f}",
                f"{train_acc:.4f}",
                f"{val_loss:.4f}",
                f"{val_acc:.4f}",
                f"{val_f1:.4f}",
            ]
        )
        # Mensaje de progreso en consola
        print(
            f"[{epoch+1:03d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
        )

        # Guardar mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
    # Cerrar el CSV de métricas y el writer de TensorBoard
    metrics_file.close()
    writer.close()
    print(
        f"✅ Entrenamiento finalizado. Mejor val_acc={best_acc:.4f} (épo. {best_epoch})"
    )

    # Evaluar el mejor modelo en el conjunto de prueba
    test_items = [r for r in items if r["split"] == "test"]
    if test_items:
        # Preparar datos de test
        test_ds = GenreDataset(test_items, "test", augment=False)
        test_dl = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config["training"]["num_workers"],
        )
        # Cargar el mejor modelo guardado
        model.load_state_dict(
            torch.load(ckpt_dir / "best_model.pt", map_location=device)
        )
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(device)
                logits = model(xb)
                preds += logits.argmax(1).cpu().tolist()
                gts += yb.cpu().tolist()
        # Preparar nombres de clases en orden índice para el reporte
        labels_map = {r["label_idx"]: r["label"] for r in items}
        class_names = [labels_map[i] for i in sorted(labels_map.keys())]
        # Generar y guardar reporte de clasificación
        report = classification_report(gts, preds, target_names=class_names, digits=4)
        print(report)
        with open(run_dir / "classification_report.txt", "w") as f:
            f.write(report)
        # Matriz de confusión
        cm = confusion_matrix(gts, preds)

        def plot_confusion_matrix(cm, class_names, out_path):
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111)
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_title("Matriz de confusión")
            ax.set_xticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticks(range(len(class_names)))
            ax.set_yticklabels(class_names)
            # Escribir valores en cada celda
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j, i, str(cm[i, j]), ha="center", va="center", color="black"
                    )
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

        plot_confusion_matrix(cm, class_names, run_dir / "confusion_matrix.png")
        print(
            f"✅ Reporte de clasificación guardado en {run_dir/'classification_report.txt'}"
        )
        print(f"✅ Matriz de confusión guardada en {run_dir/'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
