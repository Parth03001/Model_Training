"""Project CRUD API routes."""

import json
import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import select, func

from app.api.deps import DbSession
from app.models.project import Project
from app.models.image import Image
from app.schemas.project import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectListResponse,
)

router = APIRouter(redirect_slashes=False)


def _project_to_response(project: Project, image_count: int = 0, annotated_count: int = 0) -> ProjectResponse:
    classes = json.loads(project.classes) if project.classes else []
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        task_type=project.task_type,
        classes=classes,
        image_count=image_count,
        annotated_count=annotated_count,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(data: ProjectCreate, db: DbSession):
    project = Project(
        name=data.name,
        description=data.description,
        task_type=data.task_type,
        classes=json.dumps(data.classes),
    )
    db.add(project)
    await db.flush()
    await db.refresh(project)
    return _project_to_response(project)


@router.get("", response_model=ProjectListResponse)
async def list_projects(db: DbSession, skip: int = 0, limit: int = 50):
    result = await db.execute(select(Project).offset(skip).limit(limit))
    projects = result.scalars().all()

    count_result = await db.execute(select(func.count(Project.id)))
    total = count_result.scalar() or 0

    items = []
    for p in projects:
        img_count_q = await db.execute(select(func.count(Image.id)).where(Image.project_id == p.id))
        img_count = img_count_q.scalar() or 0
        annotated_q = await db.execute(
            select(func.count(Image.id)).where(Image.project_id == p.id, Image.status == "annotated")
        )
        annotated_count = annotated_q.scalar() or 0
        items.append(_project_to_response(p, img_count, annotated_count))

    return ProjectListResponse(projects=items, total=total)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: uuid.UUID, db: DbSession):
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    img_count_q = await db.execute(select(func.count(Image.id)).where(Image.project_id == project_id))
    img_count = img_count_q.scalar() or 0
    return _project_to_response(project, img_count)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: uuid.UUID, data: ProjectUpdate, db: DbSession):
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if data.name is not None:
        project.name = data.name
    if data.description is not None:
        project.description = data.description
    if data.task_type is not None:
        project.task_type = data.task_type
    if data.classes is not None:
        project.classes = json.dumps(data.classes)

    await db.flush()
    await db.refresh(project)
    return _project_to_response(project)


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: uuid.UUID, db: DbSession):
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    await db.delete(project)
