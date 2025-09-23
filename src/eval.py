# src/eval.py
import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import matplotlib.pyplot as plt

from train import load_items_from_csv
from data.dataset import GenreDataset
from models.cnn_baseline import CNNBaseline


def plot_confusion_matrix(cm, class_names, out_path):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Matriz de confusión")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación independiente de checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default="configs/gtzan_cnn.yaml", help="Ruta YAML"
    )
    parser.add_argument(
        "--run_dir", type=str, required=True, help="Directorio de run con checkpoints/"
    )
    parser.add_argument(
        "--ckpt", type=str, default="", help="Ruta a un checkpoint concreto (opcional)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Directorio de salida (por defecto, el run_dir)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    csv_path = config["dataset"]["csv_path"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    num_classes = config["model"]["num_classes"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset de test
    items = load_items_from_csv(csv_path)
    test_items = [r for r in items if r["split"] == "test"]
    if not test_items:
        print("No hay split 'test' en el CSV. Abortando.")
        return

    test_ds = GenreDataset(items, "test", config=config, augment=False)
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Modelo y checkpoint
    model = CNNBaseline(n_classes=num_classes).to(device)
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No se encontró checkpoint: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            logits = model(xb)
            preds += logits.argmax(1).cpu().tolist()
            gts += yb.cpu().tolist()

    # Nombres de clases
    labels_map = {r["label_idx"]: r["label"] for r in items}
    class_names = [labels_map[i] for i in sorted(labels_map.keys())]

    # Métricas
    report = classification_report(gts, preds, target_names=class_names, digits=4)
    acc = accuracy_score(gts, preds)
    f1m = f1_score(gts, preds, average="macro")
    print("\n=== RESULTADOS TEST ===")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1m:.4f}")
    print(report)

    # Guardar
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(gts, preds)
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")
    print(f"✅ Guardado reporte en: {out_dir/'classification_report.txt'}")
    print(f"✅ Guardada matriz de confusión en: {out_dir/'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
