from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from time import time

import pandas as pd
import torch
from matplotlib import pyplot as plt

from .utils import ensure_dir


@dataclass
class TrainerConfig:
    epochs: int
    label_smoothing: float
    early_stopping_patience: int
    use_amp: bool


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, scaler=None, device=None) -> dict:
    checkpoint = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint


def run_epoch(model, loader, criterion, optimizer, device, training: bool, scaler, use_amp: bool):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += targets.size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train_model(
    experiment_name: str,
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    device,
    config: TrainerConfig,
    checkpoint_dir: str | Path,
    results_dir: str | Path,
    log_dir: str | Path,
    resume_path: str | Path | None = None,
):
    checkpoint_dir = ensure_dir(checkpoint_dir)
    results_dir = ensure_dir(results_dir)
    log_dir = ensure_dir(log_dir)
    log_path = log_dir / f"{experiment_name}.log"
    best_accuracy = 0.0
    best_epoch = -1
    epochs_without_improvement = 0
    history = []
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")
    start_epoch = 1
    start = time()
    if resume_path is not None:
        checkpoint = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_accuracy = checkpoint.get("best_accuracy", 0.0)
        best_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch", -1))
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        history_path = results_dir / f"{experiment_name}_history.csv"
        if history_path.exists():
            history_df = pd.read_csv(history_path)
            history = history_df[history_df["epoch"] < start_epoch].to_dict("records")
        log_message(
            log_path,
            (
                f"[{timestamp()}] experiment={experiment_name} status=resumed "
                f"resume_path={resume_path} start_epoch={start_epoch} best_epoch={best_epoch} "
                f"best_test_accuracy={best_accuracy:.4f}"
            ),
        )
    else:
        log_message(log_path, f"[{timestamp()}] experiment={experiment_name} status=started device={device}")

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start = time()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True,
            scaler=scaler,
            use_amp=config.use_amp,
        )
        eval_metrics = run_epoch(
            model,
            test_loader,
            criterion,
            optimizer,
            device,
            training=False,
            scaler=None,
            use_amp=config.use_amp,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": eval_metrics["loss"],
            "test_accuracy": eval_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time() - epoch_start,
        }
        history.append(row)
        log_message(
            log_path,
            (
                f"[{timestamp()}] experiment={experiment_name} epoch={epoch}/{config.epochs} "
                f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
                f"test_loss={row['test_loss']:.4f} test_acc={row['test_accuracy']:.4f} "
                f"lr={row['lr']:.6f} epoch_seconds={row['epoch_seconds']:.2f}"
            ),
        )

        if eval_metrics["accuracy"] > best_accuracy:
            best_accuracy = eval_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch": epoch,
                    "best_accuracy": best_accuracy,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
            },
            checkpoint_dir / "last.pt",
        )

        if epochs_without_improvement >= config.early_stopping_patience:
            log_message(
                log_path,
                f"[{timestamp()}] experiment={experiment_name} status=early_stop epoch={epoch}",
            )
            break

    elapsed_seconds = time() - start
    history_df = pd.DataFrame(history)
    history_df.to_csv(results_dir / f"{experiment_name}_history.csv", index=False)
    plot_history(history_df, results_dir / f"{experiment_name}_curves.png")
    log_message(
        log_path,
        (
            f"[{timestamp()}] experiment={experiment_name} status=finished best_epoch={best_epoch} "
            f"best_test_accuracy={best_accuracy:.4f} elapsed_seconds={elapsed_seconds:.2f}"
        ),
    )
    return {
        "experiment": experiment_name,
        "best_epoch": best_epoch,
        "best_test_accuracy": best_accuracy,
        "final_train_accuracy": history_df.iloc[-1]["train_accuracy"],
        "elapsed_seconds": elapsed_seconds,
        "epochs_ran": len(history_df),
    }


def plot_history(history_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["test_loss"], label="test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], label="train")
    axes[1].plot(history_df["epoch"], history_df["test_accuracy"], label="test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_message(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(message + "\n")
