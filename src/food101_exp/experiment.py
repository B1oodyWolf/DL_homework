from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import yaml

from .data import Food101Paths, create_dataloaders
from .models import build_model, build_optimizer
from .train import TrainerConfig, train_model
from .utils import ensure_dir, resolve_device, save_json, set_seed


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_resume_path(experiment_name: str, resume: str | None) -> Path | None:
    if not resume or resume == "none":
        return None
    checkpoint_dir = Path("checkpoints") / experiment_name
    if resume == "last":
        return checkpoint_dir / "last.pt"
    if resume == "best":
        return checkpoint_dir / "best.pt"
    return Path(resume)


def run_experiments(
    config_path: str | Path,
    experiment_name: str | None = None,
    resume: str | None = None,
    epochs_override: int | None = None,
) -> pd.DataFrame:
    config = load_config(config_path)
    if epochs_override is not None:
        config["training"]["epochs"] = epochs_override
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    data_config = config["data"]
    training_config = config["training"]
    experiments = config["experiments"]
    names = [experiment_name] if experiment_name else list(experiments)

    paths = Food101Paths(
        images_root=Path(data_config["images_root"]),
        train_list=Path(data_config["train_list"]),
        test_list=Path(data_config["test_list"]),
    )
    train_dataset, _, train_loader, test_loader = create_dataloaders(
        paths=paths,
        image_size=data_config["image_size"],
        batch_size=training_config["batch_size"],
        num_workers=config["num_workers"],
        max_train_samples=data_config.get("max_train_samples"),
        max_test_samples=data_config.get("max_test_samples"),
        pin_memory=training_config.get("pin_memory", True),
        persistent_workers=training_config.get("persistent_workers", True),
        prefetch_factor=training_config.get("prefetch_factor", 4),
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    summary = []
    for name in names:
        experiment = experiments[name]
        experiment_epochs = epochs_override or experiment.get("epochs", training_config["epochs"])
        model = build_model(
            architecture=experiment["architecture"],
            mode=experiment["mode"],
            num_classes=len(train_dataset.class_to_idx),
        ).to(device)
        optimizer = build_optimizer(
            model=model,
            architecture=experiment["architecture"],
            mode=experiment["mode"],
            lr=experiment["lr"],
            backbone_lr=experiment["backbone_lr"],
            weight_decay=training_config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=experiment_epochs)
        resume_path = resolve_resume_path(name, resume)
        if resume_path is not None and not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        metrics = train_model(
            experiment_name=name,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=TrainerConfig(
                epochs=experiment_epochs,
                label_smoothing=training_config["label_smoothing"],
                early_stopping_patience=training_config["early_stopping_patience"],
                use_amp=training_config.get("use_amp", True),
            ),
            checkpoint_dir=ensure_dir(Path("checkpoints") / name),
            results_dir=ensure_dir("results"),
            log_dir=ensure_dir("logs"),
            resume_path=resume_path,
        )
        metrics["architecture"] = experiment["architecture"]
        metrics["mode"] = experiment["mode"]
        metrics["configured_epochs"] = experiment_epochs
        summary.append(metrics)

    summary_df = pd.DataFrame(summary).sort_values(by="best_test_accuracy", ascending=False)
    summary_df.to_csv("results/summary.csv", index=False)
    save_json({"device": str(device), "experiments": names}, "results/run_metadata.json")
    return summary_df
