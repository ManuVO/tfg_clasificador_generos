"""Training entry-point with Sprint 1 enhancements."""

from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import GenreDataset
from features.melspectrogram import mel_spec
from models.cnn_baseline import CNNBaseline


@dataclass
class RunningStats:
    count: int = 0
    sum_: float = 0.0
    sumsq: float = 0.0

    def update(self, array: np.ndarray) -> None:
        arr = np.asarray(array, dtype=np.float64)
        self.count += arr.size
        self.sum_ += float(arr.sum())
        self.sumsq += float(np.square(arr).sum())

    def to_dict(self) -> Dict[str, float]:
        if self.count == 0:
            return {"mean": 0.0, "std": 1.0, "count": 0}
        mean = self.sum_ / self.count
        variance = max(self.sumsq / self.count - mean**2, 0.0)
        std = float(np.sqrt(variance))
        return {"mean": float(mean), "std": std, "count": int(self.count)}


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric: Optional[float] = None
        self.bad_epochs = 0

    def step(self, metric: float) -> bool:
        if self.best_metric is None or metric < self.best_metric - self.min_delta:
            self.best_metric = metric
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def load_items_from_csv(csv_path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label_idx"] = int(row["label_idx"])
            if row.get("segment_index") is not None and row["segment_index"] != "":
                row["segment_index"] = int(row["segment_index"])
            for key in ("start_sec", "end_sec", "duration_sec"):
                if row.get(key) not in (None, ""):
                    row[key] = float(row[key])
            items.append(row)
    return items


def compute_norm_stats(train_items: Iterable[Dict], config: Dict) -> Dict[str, float]:
    stats = RunningStats()
    audio_cfg = config.get("audio", {})
    n_fft = int(audio_cfg.get("n_fft", 2048))
    hop_length = int(audio_cfg.get("hop_length", 512))
    n_mels = int(audio_cfg.get("n_mels", 128))

    for item in train_items:
        waveform, sr = sf.read(item["filepath"], dtype="float32")
        mel_db = mel_spec(waveform, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        stats.update(mel_db)

    return stats.to_dict()


def ensure_norm_stats(train_items: List[Dict], config: Dict) -> Dict[str, float]:
    dataset_cfg = config.get("dataset", {})
    norm_path = dataset_cfg.get("norm_stats_path")
    if not norm_path:
        return {}

    norm_path = Path(norm_path)
    if not norm_path.is_absolute():
        norm_path = Path.cwd() / norm_path

    recompute = bool(dataset_cfg.get("recompute_norm_stats", False))

    if norm_path.exists() and not recompute:
        with open(norm_path, "r", encoding="utf-8") as f:
            return json.load(f)

    norm_path.parent.mkdir(parents=True, exist_ok=True)
    stats = compute_norm_stats(train_items, config)
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def metrics_to_rows(name: str, metrics: Dict) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    val_metrics: Optional[Dict] = None
    test_metrics: Optional[Dict] = None

    if "best_checkpoint" in metrics:
        val_metrics = {
            "accuracy": metrics["best_checkpoint"].get("val_accuracy"),
            "macro_f1": metrics["best_checkpoint"].get("val_f1_macro"),
        }
    elif "val" in metrics:
        val_metrics = metrics.get("val")

    if "test" in metrics:
        test_metrics = metrics["test"]

    def _format_row(split: str, values: Dict[str, Optional[float]]) -> Dict[str, str]:
        acc = values.get("accuracy")
        f1 = values.get("macro_f1")
        acc = float(acc) if acc is not None else float("nan")
        f1 = float(f1) if f1 is not None else float("nan")
        return {
            "run": name,
            "split": split,
            "accuracy": f"{acc:.4f}",
            "macro_f1": f"{f1:.4f}",
        }

    if val_metrics is not None:
        rows.append(_format_row("val", val_metrics))
    if test_metrics is not None:
        rows.append(_format_row("test", test_metrics))

    return rows


def update_comparison_table(
    comparison_path: Optional[str],
    baseline_path: Optional[str],
    current_metrics: Dict,
) -> None:
    if not comparison_path:
        return

    comp_path = Path(comparison_path)
    if not comp_path.is_absolute():
        comp_path = Path.cwd() / comp_path
    comp_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []

    if baseline_path:
        baseline = Path(baseline_path)
        if not baseline.is_absolute():
            baseline = Path.cwd() / baseline
        if baseline.exists():
            with open(baseline, "r", encoding="utf-8") as f:
                baseline_metrics = json.load(f)
            name = baseline_metrics.get("experiment_name", "baseline")
            rows.extend(metrics_to_rows(name, baseline_metrics))

    name = current_metrics.get("experiment_name", "current")
    rows.extend(metrics_to_rows(name, current_metrics))

    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "split", "accuracy", "macro_f1"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(config_path: str = "configs/gtzan_cnn.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("training", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    training_cfg = config.get("training", {})

    csv_path = config["dataset"]["csv_path"]
    epochs = int(training_cfg.get("epochs", 1))
    batch_size = int(training_cfg.get("batch_size", 1))
    lr = float(training_cfg.get("learning_rate", 1e-3))
    num_classes = int(config["model"]["num_classes"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    progress_bar_enabled = bool(training_cfg.get("progress_bar", False))

    items = load_items_from_csv(csv_path)
    train_items = [row for row in items if row["split"] == "train"]
    val_items = [row for row in items if row["split"] == "val"]
    test_items = [row for row in items if row["split"] == "test"]

    norm_stats = ensure_norm_stats(train_items, config)

    train_ds = GenreDataset(items, "train", config=config, augment=True)
    val_ds = GenreDataset(items, "val", config=config, augment=False)

    num_workers = int(training_cfg.get("num_workers", 0))

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model_name = config.get("model", {}).get("name", "cnn_baseline")
    if model_name != "cnn_baseline":
        raise NotImplementedError(f"Modelo {model_name} no soportado todavía")
    model = CNNBaseline(n_classes=num_classes).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    scheduler_cfg = training_cfg.get("lr_scheduler", {})
    scheduler_kwargs = {
        "mode": "min",
        "factor": float(scheduler_cfg.get("factor", 0.5)),
        "patience": int(scheduler_cfg.get("patience", 4)),
        "threshold": float(scheduler_cfg.get("threshold", 0.001)),
        "min_lr": float(scheduler_cfg.get("min_lr", 1e-6)),
    }

    # torch < 1.1.0 no acepta el argumento ``verbose``; añadimos la clave
    # solo cuando el constructor la expone para mantener compatibilidad.
    try:
        import inspect

        signature = inspect.signature(optim.lr_scheduler.ReduceLROnPlateau.__init__)
        if "verbose" in signature.parameters:
            scheduler_kwargs["verbose"] = bool(scheduler_cfg.get("verbose", False))
    except (ValueError, TypeError):
        # Fallback silencioso si la introspección falla (p. ej., compilado en C++).
        pass

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, **scheduler_kwargs)

    early_cfg = training_cfg.get("early_stopping", {})
    early_stopper = EarlyStopping(
        patience=int(early_cfg.get("patience", 8)),
        min_delta=float(early_cfg.get("min_delta", 0.0)),
    )

    experiment_name = config.get("experiment", {}).get("name", Path(config_path).stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("experiments") / experiment_name / f"run_{timestamp}"
    logs_dir = run_dir / "logs"
    ckpt_dir = run_dir / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, run_dir / "config.yaml")

    writer = SummaryWriter(str(logs_dir))

    metrics_csv = open(run_dir / "metrics.csv", "w", newline="")
    metrics_writer = csv.writer(metrics_csv)
    metrics_writer.writerow(
        [
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "val_f1_macro",
            "learning_rate",
        ]
    )

    history: List[Dict[str, float]] = []
    best_checkpoint = {
        "val_loss": float("inf"),
        "epoch": 0,
        "path": str(ckpt_dir / "best_model.pt"),
    }

    early_stop_triggered = False

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        train_iterator: Iterable = train_dl
        if progress_bar_enabled:
            train_iterator = tqdm(
                train_dl,
                desc=f"Época {epoch}/{epochs}",
                unit="batch",
                leave=False,
            )

        for xb, yb in train_iterator:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            batch = yb.size(0)
            total_loss += loss.item() * batch
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_samples += batch

            if progress_bar_enabled:
                train_iterator.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{opt.param_groups[0]['lr']:.2e}",
                )

        if progress_bar_enabled:
            train_iterator.close()

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)

        model.eval()
        val_preds: List[int] = []
        val_gts: List[int] = []
        val_loss_sum = 0.0

        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                val_loss_sum += loss.item() * yb.size(0)
                val_preds.extend(logits.argmax(1).cpu().tolist())
                val_gts.extend(yb.cpu().tolist())

        val_loss = val_loss_sum / max(len(val_gts), 1)
        val_acc = accuracy_score(val_gts, val_preds) if val_gts else 0.0
        val_f1 = f1_score(val_gts, val_preds, average="macro") if val_gts else 0.0
        current_lr = opt.param_groups[0]["lr"]

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("val/F1_macro", val_f1, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        metrics_writer.writerow(
            [
                epoch,
                f"{train_loss:.4f}",
                f"{train_acc:.4f}",
                f"{val_loss:.4f}",
                f"{val_acc:.4f}",
                f"{val_f1:.4f}",
                f"{current_lr:.6f}",
            ]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1_macro": val_f1,
                "learning_rate": current_lr,
            }
        )

        print(
            f"[{epoch:03d}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
        )

        if val_loss < best_checkpoint["val_loss"] - early_stopper.min_delta:
            best_checkpoint.update(
                {
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_f1_macro": val_f1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "learning_rate": current_lr,
                    "epoch": epoch,
                }
            )
            torch.save(model.state_dict(), best_checkpoint["path"])

        scheduler.step(val_loss)

        if early_stopper.step(val_loss):
            early_stop_triggered = True
            print(f"⏹️  Deteniendo entrenamiento por early stopping en la época {epoch}.")
            break

    metrics_csv.close()
    writer.close()

    print(
        "✅ Entrenamiento finalizado. "
        f"Mejor val_loss={best_checkpoint['val_loss']:.4f} (época {best_checkpoint['epoch']})."
    )

    metrics_summary: Dict = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "config_path": str(Path(config_path).resolve()),
        "epochs_requested": epochs,
        "epochs_trained": len(history),
        "early_stopped": early_stop_triggered,
        "best_checkpoint": best_checkpoint,
        "epochs": history,
        "norm_stats": norm_stats,
    }

    if early_stop_triggered:
        metrics_summary["early_stopped_at"] = history[-1]["epoch"] if history else 0

    if test_items:
        test_ds = GenreDataset(items, "test", config=config, augment=False)
        test_dl = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        model.load_state_dict(torch.load(best_checkpoint["path"], map_location=device))
        model.eval()

        test_preds: List[int] = []
        test_gts: List[int] = []

        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(device)
                logits = model(xb)
                test_preds.extend(logits.argmax(1).cpu().tolist())
                test_gts.extend(yb.cpu().tolist())

        test_acc = accuracy_score(test_gts, test_preds) if test_gts else 0.0
        test_f1 = f1_score(test_gts, test_preds, average="macro") if test_gts else 0.0

        metrics_summary["test"] = {"accuracy": test_acc, "macro_f1": test_f1}

        labels_map = {row["label_idx"]: row["label"] for row in items}
        class_names = [labels_map[idx] for idx in sorted(labels_map.keys())]
        report = classification_report(test_gts, test_preds, target_names=class_names, digits=4)

        with open(run_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        cm = confusion_matrix(test_gts, test_preds)

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
        fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

        print(f"✅ Reporte de clasificación guardado en {run_dir/'classification_report.txt'}")
        print(f"✅ Matriz de confusión guardada en {run_dir/'confusion_matrix.png'}")

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    reports_cfg = config.get("reports", {})
    update_comparison_table(
        reports_cfg.get("comparison_table_path"),
        reports_cfg.get("baseline_metrics_path"),
        metrics_summary,
    )


if __name__ == "__main__":
    main()
