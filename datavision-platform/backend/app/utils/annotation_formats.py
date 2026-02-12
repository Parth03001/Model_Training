"""Annotation format conversion utilities (YOLO, COCO, VOC, CSV)."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path


def to_yolo_format(annotations: list[dict], class_names: list[str]) -> str:
    """
    Convert annotations to YOLO format.

    YOLO format: <class_idx> <cx> <cy> <w> <h>
    All values normalized to [0, 1].
    """
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    lines = []
    for ann in annotations:
        cls_idx = class_to_idx.get(ann["class_name"], 0)
        lines.append(f"{cls_idx} {ann['bbox_x']:.6f} {ann['bbox_y']:.6f} {ann['bbox_w']:.6f} {ann['bbox_h']:.6f}")
    return "\n".join(lines)


def to_coco_format(
    images: list[dict],
    annotations: dict[str, list[dict]],
    class_names: list[str],
) -> dict:
    """Convert to COCO JSON format."""
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(class_names)],
    }
    class_to_id = {name: i + 1 for i, name in enumerate(class_names)}
    ann_id = 1

    for img in images:
        coco["images"].append({
            "id": img["id"],
            "file_name": img["filename"],
            "width": img["width"],
            "height": img["height"],
        })
        for ann in annotations.get(img["id"], []):
            if ann.get("bbox_x") is not None:
                # Convert from center format to top-left format
                x = (ann["bbox_x"] - ann["bbox_w"] / 2) * img["width"]
                y = (ann["bbox_y"] - ann["bbox_h"] / 2) * img["height"]
                w = ann["bbox_w"] * img["width"]
                h = ann["bbox_h"] * img["height"]
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img["id"],
                    "category_id": class_to_id.get(ann["class_name"], 1),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1

    return coco


def to_voc_format(
    image: dict,
    annotations: list[dict],
) -> str:
    """Convert annotations for a single image to Pascal VOC XML format."""
    root = ET.Element("annotation")

    ET.SubElement(root, "filename").text = image["filename"]
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image["width"])
    ET.SubElement(size, "height").text = str(image["height"])
    ET.SubElement(size, "depth").text = "3"

    for ann in annotations:
        if ann.get("bbox_x") is None:
            continue

        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = ann["class_name"]
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        x1 = (ann["bbox_x"] - ann["bbox_w"] / 2) * image["width"]
        y1 = (ann["bbox_y"] - ann["bbox_h"] / 2) * image["height"]
        x2 = (ann["bbox_x"] + ann["bbox_w"] / 2) * image["width"]
        y2 = (ann["bbox_y"] + ann["bbox_h"] / 2) * image["height"]

        ET.SubElement(bndbox, "xmin").text = str(int(x1))
        ET.SubElement(bndbox, "ymin").text = str(int(y1))
        ET.SubElement(bndbox, "xmax").text = str(int(x2))
        ET.SubElement(bndbox, "ymax").text = str(int(y2))

    return ET.tostring(root, encoding="unicode")
