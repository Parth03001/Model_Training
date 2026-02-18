"""Celery tasks for model training — run on GPU workers."""

import json
from datetime import datetime
from loguru import logger

from app.tasks.celery_app import celery_app


@celery_app.task(name="app.tasks.training_tasks.run_training_job", bind=True)
def run_training_job(self, job_id: str):
    """
    Execute a training job. Dispatches to the appropriate trainer
    based on the job's task_type and model_architecture.
    """
    from app.database import async_session
    from app.models.training_job import TrainingJob
    from app.models.image import Image
    from app.models.annotation import Annotation
    from app.models.trained_model import TrainedModel
    from app.models.project import Project
    import asyncio
    from sqlalchemy import select

    async def _run():
        async with async_session() as db:
            # Load job config
            result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
            job = result.scalar_one_or_none()
            if not job:
                return {"error": "Job not found"}

            job.status = "preparing"
            job.started_at = datetime.utcnow()
            await db.commit()

            # Load project
            proj_result = await db.execute(select(Project).where(Project.id == job.project_id))
            project = proj_result.scalar_one_or_none()
            class_names = json.loads(project.classes) if project.classes else []

            # Load images and annotations
            img_result = await db.execute(select(Image).where(Image.project_id == job.project_id))
            images = img_result.scalars().all()

            image_records = []
            annotations_map = {}
            for img in images:
                image_records.append({
                    "id": str(img.id),
                    "filepath": img.filepath,
                    "width": img.width,
                    "height": img.height,
                    "class_name": None,  # Will be filled for classification
                })
                ann_result = await db.execute(select(Annotation).where(Annotation.image_id == img.id))
                anns = ann_result.scalars().all()
                annotations_map[str(img.id)] = [
                    {
                        "class_name": a.class_name,
                        "bbox_x": a.bbox_x,
                        "bbox_y": a.bbox_y,
                        "bbox_w": a.bbox_w,
                        "bbox_h": a.bbox_h,
                    }
                    for a in anns
                ]
                # For classification, use first annotation class
                if anns:
                    image_records[-1]["class_name"] = anns[0].class_name

            job.status = "training"
            await db.commit()

            # Dispatch to appropriate trainer
            try:
                if job.task_type in ("detection", "segmentation"):
                    from app.services.training.yolo_trainer import prepare_yolo_dataset, train

                    data_yaml = prepare_yolo_dataset(
                        str(job.project_id), image_records,
                        annotations_map, class_names,
                    )
                    augmentation = json.loads(job.augmentation_config) if job.augmentation_config else None

                    result = train(
                        job_id=job_id,
                        data_yaml=str(data_yaml),
                        model_architecture=job.model_architecture,
                        epochs=job.epochs,
                        batch_size=job.batch_size,
                        img_size=job.img_size,
                        learning_rate=job.learning_rate,
                        patience=job.patience,
                        optimizer=job.optimizer,
                        augmentation=augmentation,
                    )

                elif job.task_type == "classification":
                    from app.services.training.vit_trainer import prepare_classification_dataset, train

                    dataset_config = prepare_classification_dataset(
                        str(job.project_id), image_records, class_names,
                        img_size=job.img_size,
                    )
                    result = train(
                        job_id=job_id,
                        dataset_config=dataset_config,
                        model_architecture=job.model_architecture,
                        epochs=job.epochs,
                        batch_size=job.batch_size,
                        img_size=job.img_size,
                        learning_rate=job.learning_rate,
                        patience=job.patience,
                        optimizer=job.optimizer,
                    )
                else:
                    raise ValueError(f"Unsupported task type: {job.task_type}")

                # Update job status
                job.status = "completed"
                job.model_path = result["model_path"]
                job.best_metrics = json.dumps(result.get("metrics", {}))
                job.completed_at = datetime.utcnow()

                # Register trained model
                metrics = result.get("metrics", {})
                trained_model = TrainedModel(
                    project_id=job.project_id,
                    training_job_id=job.id,
                    name=f"{job.name} - {job.model_architecture}",
                    architecture=job.model_architecture,
                    task_type=job.task_type,
                    model_path=result["model_path"],
                    num_classes=len(class_names),
                    class_names=json.dumps(class_names),
                    accuracy=metrics.get("accuracy"),
                    map50=metrics.get("map50"),
                    map50_95=metrics.get("map50_95"),
                    precision_val=metrics.get("precision"),
                    recall_val=metrics.get("recall"),
                )
                db.add(trained_model)
                await db.commit()

                logger.info(f"Training job {job_id} completed successfully")
                return result

            except Exception as e:
                job.status = "failed"
                job.best_metrics = json.dumps({"error": str(e)})
                await db.commit()
                logger.error(f"Training job {job_id} failed: {e}")
                raise

    return asyncio.run(_run())


@celery_app.task(name="app.tasks.training_tasks.run_model_export", bind=True)
def run_model_export(self, model_id: str, formats: list[str], img_size: int, quantize: bool):
    """Export a trained model to various formats (ONNX, TFLite, TorchScript, CoreML)."""
    from app.database import async_session
    from app.models.trained_model import TrainedModel
    import asyncio
    from sqlalchemy import select

    async def _export():
        async with async_session() as db:
            result = await db.execute(select(TrainedModel).where(TrainedModel.id == model_id))
            model = result.scalar_one_or_none()
            if not model:
                return {"error": "Model not found"}

            export_results = {}

            if model.architecture.startswith("yolo") or model.architecture.startswith("rt_detr"):
                from ultralytics import YOLO
                yolo = YOLO(model.model_path)

                for fmt in formats:
                    try:
                        export_path = yolo.export(format=fmt, imgsz=img_size)
                        export_results[fmt] = str(export_path)
                    except Exception as e:
                        export_results[fmt] = f"Failed: {e}"

            elif model.architecture.startswith("vit") or model.architecture.startswith("efficientnet"):
                import torch
                from pathlib import Path

                checkpoint = torch.load(model.model_path, map_location="cpu")
                from app.services.training.vit_trainer import _build_model
                net = _build_model(model.architecture, model.num_classes)
                net.load_state_dict(checkpoint["model_state_dict"])
                net.eval()

                dummy_input = torch.randn(1, 3, img_size, img_size)
                export_dir = Path(model.model_path).parent

                for fmt in formats:
                    try:
                        if fmt == "onnx":
                            path = export_dir / f"model.onnx"
                            torch.onnx.export(net, dummy_input, str(path), opset_version=17)
                            export_results["onnx"] = str(path)
                        elif fmt == "torchscript":
                            path = export_dir / f"model.torchscript"
                            traced = torch.jit.trace(net, dummy_input)
                            traced.save(str(path))
                            export_results["torchscript"] = str(path)
                    except Exception as e:
                        export_results[fmt] = f"Failed: {e}"

            return export_results

    return asyncio.run(_export())
