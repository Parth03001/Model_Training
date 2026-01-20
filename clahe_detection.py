"""
CLAHE Enhanced YOLOv11 Detection Script
Compares detection results before and after CLAHE preprocessing
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse


def apply_clahe(image, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to image
    
    Args:
        image: Input BGR image
        clip_limit: Threshold for contrast limiting (1.0-4.0, default 2.5)
        tile_grid_size: Size of grid for histogram equalization (default 8x8)
    
    Returns:
        CLAHE enhanced image
    """
    # Convert BGR to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE to L channel only
    l_channel_clahe = clahe.apply(l_channel)
    
    # Merge channels back
    lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])
    
    # Convert back to BGR
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return enhanced_image


def draw_detections(image, results, label=""):
    """
    Draw bounding boxes and labels on image
    
    Args:
        image: Input image
        results: YOLO detection results
        label: Optional label to add to image
    
    Returns:
        Image with drawn bounding boxes
    """
    annotated_image = image.copy()
    
    # Get detection boxes
    boxes = results[0].boxes
    
    if len(boxes) > 0:
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label text
            text = f"{class_name}: {confidence:.2f}"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                annotated_image,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated_image,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
    
    # Add label to image (BEFORE/AFTER)
    if label:
        cv2.putText(
            annotated_image,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )
    
    # Add detection count
    detection_count = len(boxes)
    count_text = f"Detections: {detection_count}"
    cv2.putText(
        annotated_image,
        count_text,
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )
    
    return annotated_image


def create_comparison(image_before, image_after, title="CLAHE Comparison"):
    """
    Create side-by-side comparison of before and after images
    
    Args:
        image_before: Original image with detections
        image_after: CLAHE enhanced image with detections
        title: Title for the comparison
    
    Returns:
        Combined comparison image
    """
    # Get dimensions
    h1, w1 = image_before.shape[:2]
    h2, w2 = image_after.shape[:2]
    
    # Create canvas with space for title
    title_height = 50
    max_height = max(h1, h2)
    total_width = w1 + w2
    
    # Create white canvas
    canvas = np.ones((max_height + title_height, total_width, 3), dtype=np.uint8) * 255
    
    # Add title
    cv2.putText(
        canvas,
        title,
        (total_width // 2 - 200, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2
    )
    
    # Place images side by side
    canvas[title_height:title_height + h1, 0:w1] = image_before
    canvas[title_height:title_height + h2, w1:w1 + w2] = image_after
    
    # Draw dividing line
    cv2.line(
        canvas,
        (w1, title_height),
        (w1, title_height + max_height),
        (0, 0, 0),
        3
    )
    
    return canvas


def print_detection_stats(results_before, results_after):
    """
    Print comparison statistics
    
    Args:
        results_before: Detection results before CLAHE
        results_after: Detection results after CLAHE
    """
    boxes_before = results_before[0].boxes
    boxes_after = results_after[0].boxes
    
    print("\n" + "="*60)
    print("DETECTION COMPARISON STATISTICS")
    print("="*60)
    
    print(f"\nBEFORE CLAHE:")
    print(f"  Total Detections: {len(boxes_before)}")
    if len(boxes_before) > 0:
        confidences_before = [float(box.conf[0]) for box in boxes_before]
        print(f"  Average Confidence: {np.mean(confidences_before):.4f}")
        print(f"  Max Confidence: {max(confidences_before):.4f}")
        print(f"  Min Confidence: {min(confidences_before):.4f}")
    
    print(f"\nAFTER CLAHE:")
    print(f"  Total Detections: {len(boxes_after)}")
    if len(boxes_after) > 0:
        confidences_after = [float(box.conf[0]) for box in boxes_after]
        print(f"  Average Confidence: {np.mean(confidences_after):.4f}")
        print(f"  Max Confidence: {max(confidences_after):.4f}")
        print(f"  Min Confidence: {min(confidences_after):.4f}")
    
    print(f"\nIMPROVEMENT:")
    detection_diff = len(boxes_after) - len(boxes_before)
    print(f"  Detection Difference: {detection_diff:+d}")
    
    if len(boxes_before) > 0 and len(boxes_after) > 0:
        conf_improvement = np.mean(confidences_after) - np.mean(confidences_before)
        print(f"  Confidence Improvement: {conf_improvement:+.4f}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='CLAHE Enhanced YOLOv11 Detection')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='C:\\Users\\50014665\\Image_Classification\\connector_inspection\\yolo11s_single_stage\\weights\\best.pt', help='Path to YOLOv11 model')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--clip-limit', type=float, default=2.5, help='CLAHE clip limit (1.0-4.0)')
    parser.add_argument('--tile-size', type=int, default=8, help='CLAHE tile grid size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*60}")
    print("CLAHE ENHANCED YOLOV11 DETECTION")
    print(f"{'='*60}")
    print(f"Input Image: {args.image}")
    print(f"Model: {args.model}")
    print(f"CLAHE Clip Limit: {args.clip_limit}")
    print(f"CLAHE Tile Size: {args.tile_size}x{args.tile_size}")
    print(f"Confidence Threshold: {args.conf}")
    print(f"{'='*60}\n")
    
    # Load image
    print("Loading image...")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return
    
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Apply CLAHE
    print("Applying CLAHE preprocessing...")
    image_clahe = apply_clahe(
        image,
        clip_limit=args.clip_limit,
        tile_grid_size=(args.tile_size, args.tile_size)
    )
    
    # Load YOLO model
    print(f"Loading YOLOv11 model: {args.model}...")
    model = YOLO(args.model)
    
    # Run detection on original image
    print("Running detection on ORIGINAL image...")
    results_before = model(image, conf=args.conf, verbose=False)
    
    # Run detection on CLAHE enhanced image
    print("Running detection on CLAHE ENHANCED image...")
    results_after = model(image_clahe, conf=args.conf, verbose=False)
    
    # Draw detections
    print("Drawing bounding boxes...")
    image_before_annotated = draw_detections(image, results_before, label="BEFORE CLAHE")
    image_after_annotated = draw_detections(image_clahe, results_after, label="AFTER CLAHE")
    
    # Create comparison
    print("Creating comparison image...")
    comparison = create_comparison(image_before_annotated, image_after_annotated)
    
    # Print statistics
    print_detection_stats(results_before, results_after)
    
    # Save results
    input_filename = Path(args.image).stem
    
    # Save individual images
    cv2.imwrite(str(output_dir / f"{input_filename}_before.jpg"), image_before_annotated)
    cv2.imwrite(str(output_dir / f"{input_filename}_after_clahe.jpg"), image_after_annotated)
    cv2.imwrite(str(output_dir / f"{input_filename}_comparison.jpg"), comparison)
    
    # Save CLAHE enhanced image without annotations
    cv2.imwrite(str(output_dir / f"{input_filename}_clahe_only.jpg"), image_clahe)
    
    print("Results saved:")
    print(f"  - {output_dir / f'{input_filename}_before.jpg'}")
    print(f"  - {output_dir / f'{input_filename}_after_clahe.jpg'}")
    print(f"  - {output_dir / f'{input_filename}_comparison.jpg'}")
    print(f"  - {output_dir / f'{input_filename}_clahe_only.jpg'}")
    
    print("\n✓ Processing complete!\n")


if __name__ == "__main__":
    main()