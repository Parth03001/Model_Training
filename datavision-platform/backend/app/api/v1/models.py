"""Trained model registry API routes."""

import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File
from sqlalchemy import select

from app.api.deps import DbSession
from app.models.trained_model import TrainedModel
from app.schemas.model import TrainedModelResponse, ModelListResponse

router = APIRouter()


@router.get("/", response_model=ModelListResponse)
async def list_models(
    db: DbSession,
    project_id: uuid.UUID | None = None,
    task_type: str | None = None,
    architecture: str | None = None,
):
    query = select(TrainedModel)
    if project_id:
        query = query.where(TrainedModel.project_id == project_id)
    if task_type:
        query = query.where(TrainedModel.task_type == task_type)
    if architecture:
        query = query.where(TrainedModel.architecture == architecture)
    query = query.order_by(TrainedModel.created_at.desc())

    result = await db.execute(query)
    models = result.scalars().all()
    return ModelListResponse(
        models=[TrainedModelResponse.model_validate(m) for m in models],
        total=len(models),
    )


@router.get("/{model_id}", response_model=TrainedModelResponse)
async def get_model(model_id: uuid.UUID, db: DbSession):
    result = await db.execute(select(TrainedModel).where(TrainedModel.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return TrainedModelResponse.model_validate(model)


@router.delete("/{model_id}", status_code=204)
async def delete_model(model_id: uuid.UUID, db: DbSession):
    result = await db.execute(select(TrainedModel).where(TrainedModel.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Delete model file
    from pathlib import Path
    model_path = Path(model.model_path)
    if model_path.exists():
        model_path.unlink()

    await db.delete(model)


@router.post("/{model_id}/inference")
async def run_inference(model_id: uuid.UUID, db: DbSession, image: UploadFile = File(...)):
    """Run inference using a trained model on an uploaded image."""
    result = await db.execute(select(TrainedModel).where(TrainedModel.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Read image
    import io
    from PIL import Image as PILImage
    contents = await image.read()
    img = PILImage.open(io.BytesIO(contents))

    # Run inference based on architecture
    predictions = await _run_model_inference(model, img)
    return {"model_id": str(model_id), "predictions": predictions}


async def _run_model_inference(model: TrainedModel, img) -> list[dict]:
    """Dispatch inference to the appropriate model handler."""
    if model.architecture.startswith("yolo"):
        from ultralytics import YOLO
        yolo = YOLO(model.model_path)
        results = yolo(img, conf=0.25)
        predictions = []
        for r in results:
            for box in r.boxes:
                predictions.append({
                    "class": r.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xywhn[0].tolist(),
                })
        return predictions

    elif model.architecture.startswith("vit") or model.architecture.startswith("efficientnet"):
        import torch
        from torchvision import transforms
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded = torch.load(model.model_path, map_location=device)
        net = loaded["model"] if isinstance(loaded, dict) else loaded
        net.eval()

        transform = transforms.Compose([
            transforms.Resize((model.num_classes, model.num_classes)),  # will be fixed per model
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = net(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        import json
        class_names = json.loads(model.class_names) if model.class_names else []
        return [{
            "class": class_names[predicted.item()] if class_names else str(predicted.item()),
            "confidence": float(confidence.item()),
        }]

    return [{"error": f"Unsupported architecture: {model.architecture}"}]
