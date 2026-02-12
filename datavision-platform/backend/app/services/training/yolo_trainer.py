"""
YOLO Training Service — Trains YOLOv8/v11 models for detection and segmentation.

Supports:
    - YOLOv8 (n/s/m/l/x variants)
    - YOLOv11 (n/s/m/l variants)
    - RT-DETR (l/x variants) — transformer-based detector

Training Flow:
    1. Dataset Preparation:
       - Exports annotations to YOLO format (txt files)
       - Creates data.yaml with class names and paths
       - Applies train/val/test split

    2. Training:
       - Loads pre-trained weights from Ultralytics hub
       - Applies augmentation config
       - Trains with early stopping based on mAP50

    3. Metrics Reporting:
       - Publishes per-epoch metrics to Redis pub/sub
       - Frontend receives via WebSocket in real-time

    4. Output:
       - best.pt and last.pt saved to model directory
       - Registered in trained_models table
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

import redis
from loguru import logger

from app.config import settings


def prepare_yolo_dataset(
    project_id: str,
    images: list[dict],
    annotations: dict[str, list[dict]],
    class_names: list[str],
    train_split: float = 0.8,
    val_split: float = 0.15,
) -> Path:
    """
    Prepare YOLO-format dataset from database records.

    Args:
        project_id: Project UUID
        images: List of image records with filepath, width, height
        annotations: Dict mapping image_id -> list of annotation dicts
        class_names: Ordered list of class names
        train_split: Fraction for training
        val_split: Fraction for validation

    Returns:
        Path to data.yaml
    """
    dataset_dir = settings.model_dir / project_id / "dataset"

    # Create directory structure
    for split in ["train", "val", "test"]:
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Split images
    import random
    random.shuffle(images)
    n_train = int(len(images) * train_split)
    n_val = int(len(images) * val_split)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for split_name, split_images in splits.items():
        for img in split_images:
            # Copy image
            src = Path(img["filepath"])
            dst = dataset_dir / split_name / "images" / src.name
            if src.exists():
                shutil.copy2(src, dst)

            # Write YOLO label file
            label_path = dataset_dir / split_name / "labels" / (src.stem + ".txt")
            img_anns = annotations.get(img["id"], [])

            lines = []
            for ann in img_anns:
                if ann.get("bbox_x") is not None:
                    cls_idx = class_to_idx.get(ann["class_name"], 0)
                    lines.append(f"{cls_idx} {ann['bbox_x']} {ann['bbox_y']} {ann['bbox_w']} {ann['bbox_h']}")

            label_path.write_text("\n".join(lines))

    # Create data.yaml
    data_yaml = dataset_dir / "data.yaml"
    yaml_content = {
        "path": str(dataset_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }

    import yaml
    with open(data_yaml, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    logger.info(f"YOLO dataset prepared: {n_train} train, {n_val} val, {len(images) - n_train - n_val} test")
    return data_yaml


def train(
    job_id: str,
    data_yaml: str,
    model_architecture: str,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    learning_rate: float = 0.01,
    patience: int = 20,
    optimizer: str = "AdamW",
    augmentation: dict | None = None,
) -> dict:
    """
    Train a YOLO model.

    Args:
        job_id: Training job UUID (for progress reporting)
        data_yaml: Path to data.yaml
        model_architecture: e.g., "yolov8m", "yolov11m", "rt_detr_l"
        ... hyperparameters ...

    Returns:
        Dict with model path and final metrics
    """
    from ultralytics import YOLO

    # Map architecture to model weights
    model_map = {
        "yolov8n": "yolov8n.pt", "yolov8s": "yolov8s.pt",
        "yolov8m": "yolov8m.pt", "yolov8l": "yolov8l.pt", "yolov8x": "yolov8x.pt",
        "yolov11n": "yolo11n.pt", "yolov11s": "yolo11s.pt",
        "yolov11m": "yolo11m.pt", "yolov11l": "yolo11l.pt",
        "rt_detr_l": "rtdetr-l.pt", "rt_detr_x": "rtdetr-x.pt",
    }

    weights = model_map.get(model_architecture, "yolov8m.pt")
    model = YOLO(weights)

    # Output directory
    output_dir = settings.model_dir / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build training args
    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "lr0": learning_rate,
        "patience": patience,
        "optimizer": optimizer,
        "project": str(output_dir),
        "name": "train",
        "exist_ok": True,
        "verbose": True,
        "save": True,
        "save_period": 10,
    }

    # Apply augmentation config
    if augmentation:
        train_args.update(augmentation)

    # Setup Redis for real-time metric publishing
    r = redis.Redis.from_url(settings.redis_url)
    channel = f"training:{job_id}:metrics"

    # Custom callback to publish metrics
    def on_train_epoch_end(trainer):
        metrics = {
            "epoch": trainer.epoch,
            "train_loss": float(trainer.loss.item()) if hasattr(trainer, "loss") else 0,
            "learning_rate": float(trainer.lf(trainer.epoch)),
        }
        r.publish(channel, json.dumps(metrics))

    def on_val_end(validator):
        metrics = {
            "map50": float(validator.metrics.box.map50) if hasattr(validator.metrics, "box") else 0,
            "map50_95": float(validator.metrics.box.map) if hasattr(validator.metrics, "box") else 0,
            "precision": float(validator.metrics.box.mp) if hasattr(validator.metrics, "box") else 0,
            "recall": float(validator.metrics.box.mr) if hasattr(validator.metrics, "box") else 0,
        }
        r.publish(channel, json.dumps(metrics))

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_val_end", on_val_end)

    # Train
    logger.info(f"Starting YOLO training: {model_architecture}, {epochs} epochs")
    results = model.train(**train_args)

    # Collect final metrics
    best_model_path = output_dir / "train" / "weights" / "best.pt"
    final_metrics = {}
    if hasattr(results, "results_dict"):
        final_metrics = {k: float(v) for k, v in results.results_dict.items()}

    r.close()

    return {
        "model_path": str(best_model_path),
        "metrics": final_metrics,
        "completed_at": datetime.utcnow().isoformat(),
    }
