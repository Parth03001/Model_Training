import torch
from PIL import Image
import numpy as np
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Model, Sam2Processor
import supervision as sv
import os

# CONFIGURATION
DINO_PATH = r"datavision_hf_models\grounding-dino-base"
SAM2_PATH = r"datavision_hf_models\sam2-hiera-large"
IMAGE_PATH = r"D:\Model_Training\basket.jpg"
OUTPUT_PATH = "robust_detection_result.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    """Load Grounding DINO and SAM 2 models."""
    print(f"Loading models to {DEVICE}...")
    processor_dino = AutoProcessor.from_pretrained(DINO_PATH)
    model_dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_PATH).to(DEVICE)
    processor_sam2 = Sam2Processor.from_pretrained(SAM2_PATH)
    model_sam2 = Sam2Model.from_pretrained(SAM2_PATH).to(DEVICE)
    return processor_dino, model_dino, processor_sam2, model_sam2

def get_dino_detections(image, text_prompt, processor_dino, model_dino, box_threshold=0.12, text_threshold=0.12):
    """Run Grounding DINO with specified thresholds."""
    # Ensure prompt ends with a period for better DINO performance
    if not text_prompt.strip().endswith("."):
        text_prompt = text_prompt.strip() + " . "
        
    inputs = processor_dino(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model_dino(**inputs)
    
    width, height = image.size
    target_sizes = torch.Tensor([[height, width]]).to(DEVICE)
    results = processor_dino.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes
    )[0]
    return results

def refine_masks_sam2(image, boxes, processor_sam2, model_sam2):
    """Run SAM 2 on detected boxes with intelligent mask level selection."""
    if len(boxes) == 0:
        return []
    
    width, height = image.size
    inputs_sam2 = processor_sam2(image, input_boxes=[boxes], return_tensors="pt")
    inputs_sam2_gpu = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs_sam2.items()}
    
    with torch.no_grad():
        # multimask_output=True gives 3 levels: 0 (part), 1 (sub-part), 2 (whole)
        outputs_sam2 = model_sam2(**inputs_sam2_gpu, multimask_output=True)
    
    final_masks = []
    for i in range(len(boxes)):
        iou_scores = outputs_sam2.iou_scores[0, i].cpu().numpy()
        
        # HEURISTIC FOR ROBUSTNESS:
        # We generally want the 'whole' object (level 2).
        # We pick level 2 unless its IoU is significantly worse (>0.15) than the max.
        max_iou_idx = np.argmax(iou_scores)
        if iou_scores[2] > (iou_scores[max_iou_idx] - 0.15):
            best_mask_idx = 2
        else:
            best_mask_idx = max_iou_idx
            
        raw_mask = outputs_sam2.pred_masks[0, i, best_mask_idx]
        
        # Upsample to full resolution
        mask_full = torch.nn.functional.interpolate(
            raw_mask.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )[0, 0]
        
        # Threshold and convert to bool
        binary_mask = (mask_full > 0.0).cpu().numpy().astype(bool)
        
        # Basic mask cleanup: Remove small disconnected artifacts
        # (This helps when SAM 2 'leaks' a few pixels elsewhere)
        binary_mask_uint8 = binary_mask.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_uint8)
        if num_labels > 1:
            # Keep only the largest component
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            binary_mask = (labels == largest_label)
            
        final_masks.append(binary_mask)
        
    return np.array(final_masks)

def main(prompt):
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return

    image = Image.open(IMAGE_PATH).convert("RGB")
    processor_dino, model_dino, processor_sam2, model_sam2 = load_models()
    
    print(f"--- Starting Robust Detection ---")
    print(f"Target Prompt: '{prompt}'")
    
    # 1. First Attempt: Specific Prompt
    results = get_dino_detections(image, prompt, processor_dino, model_dino)
    
    # 2. ROBUSTNESS: Fallback logic if 0 objects found
    if len(results["boxes"]) == 0:
        print("Stage 1: No objects found. Trying decomposed prompt (splitting with '.')")
        decomposed = prompt.replace(" with ", " . ").replace(" having ", " . ").replace(" wearing ", " . ")
        results = get_dino_detections(image, decomposed, processor_dino, model_dino)
        
    if len(results["boxes"]) == 0:
        print("Stage 2: Still no objects. Trying generic class extraction...")
        # Extract first noun-like word (heuristic)
        generic = prompt.split(" ")[0]
        results = get_dino_detections(image, generic, processor_dino, model_dino)

    if len(results["boxes"]) > 0:
        boxes = results["boxes"].cpu().numpy()
        labels = results["labels"]
        scores = results["scores"].cpu().numpy()
        
        print(f"Raw detections: {len(boxes)}")
        
        # 3. NMS (Non-Maximum Suppression) to clean up duplicates
        # We use a combined class_id 0 to suppress overlaps across all detected terms
        detections = sv.Detections(
            xyxy=boxes, 
            confidence=scores, 
            class_id=np.zeros(len(boxes), dtype=int)
        )
        nms_idx = sv.nms(detections, threshold=0.5)
        
        boxes = boxes[nms_idx]
        labels = [labels[i] for i in nms_idx]
        scores = scores[nms_idx]
        print(f"After NMS cleanup: {len(boxes)} objects.")
        
        # 4. SAM 2 Refinement
        print("Generating high-quality masks with SAM 2...")
        masks = refine_masks_sam2(image, boxes, processor_sam2, model_sam2)
        
        # 5. Visualization with Supervision
        final_detections = sv.Detections(
            xyxy=boxes,
            mask=masks,
            confidence=scores
        )
        
        # Define annotators with nice styling
        mask_annotator = sv.MaskAnnotator(opacity=0.4)
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=1)
        
        annotated_image = np.array(image)
        annotated_image = mask_annotator.annotate(scene=annotated_image, detections=final_detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=final_detections)
        
        # Create labels that include the text found by DINO
        display_labels = [f"{l} {s:.2f}" for l, s in zip(labels, scores)]
        annotated_image = label_annotator.annotate(
            scene=annotated_image, 
            detections=final_detections, 
            labels=display_labels
        )
        
        # Save result
        cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"SUCCESS: Result saved to {OUTPUT_PATH}")
        print(f"Total objects visualized: {len(boxes)}")
    else:
        print("FAILURE: Could not detect any objects matching the prompt or its fallbacks.")
if __name__ == "__main__":
    # You can change this prompt to anything specific
    user_prompt = "all players on the court with yellow shirt"
    main(user_prompt)
