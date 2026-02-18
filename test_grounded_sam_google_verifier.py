import torch
from PIL import Image
import numpy as np
import cv2
from transformers import (
    AutoProcessor, 
    AutoModelForZeroShotObjectDetection, 
    Sam2Model, 
    Sam2Processor,
    AutoModelForZeroShotImageClassification # SigLIP can use this for verification
)
import supervision as sv
import os

# CONFIGURATION
DINO_PATH = r"datavision_hf_models\grounding-dino-base"
SAM2_PATH = r"datavision_hf_models\sam2-hiera-large"
SIGLIP_PATH = r"datavision_hf_models\siglip-so400m-patch14-384"
IMAGE_PATH = r"D:\Model_Training\basket.jpg"
OUTPUT_PATH = "google_verified_result.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    """Load Grounding DINO, SAM 2, and SigLIP models."""
    print(f"Loading models to {DEVICE}...")
    
    # DINO for discovery
    processor_dino = AutoProcessor.from_pretrained(DINO_PATH)
    model_dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_PATH).to(DEVICE)
    
    # SAM 2 for masking
    processor_sam2 = Sam2Processor.from_pretrained(SAM2_PATH)
    model_sam2 = Sam2Model.from_pretrained(SAM2_PATH).to(DEVICE)
    
    # SigLIP for verification (Google model)
    # SigLIP is great for zero-shot image classification/similarity
    processor_siglip = AutoProcessor.from_pretrained(SIGLIP_PATH)
    model_siglip = AutoModelForZeroShotImageClassification.from_pretrained(SIGLIP_PATH).to(DEVICE)
    
    return processor_dino, model_dino, processor_sam2, model_sam2, processor_siglip, model_siglip

def verify_with_google_siglip(image, boxes, text_prompt, processor_siglip, model_siglip, threshold=0.6):
    """
    Crop candidates and verify them using SigLIP (Google) to ensure they match the specific prompt.
    """
    if len(boxes) == 0:
        return []
    
    print(f"Verifying {len(boxes)} candidates with SigLIP (Threshold: {threshold})...")
    os.makedirs("debug_crops", exist_ok=True)
    
    verified_indices = []
    # Using more descriptive labels for better contrast
    candidate_labels = [
        f"a photo of a basketball player wearing a yellow jersey", 
        "a photo of a basketball player wearing a white jersey", 
        "a photo of a basketball backboard or rim",
        "a photo of a crowd of people"
    ]
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        # Pad the crop slightly for context (20%)
        w, h = x2 - x1, y2 - y1
        x1_p, y1_p = max(0, x1 - int(w*0.2)), max(0, y1 - int(h*0.2))
        x2_p, y2_p = min(image.width, x2 + int(w*0.2)), min(image.height, y2 + int(h*0.2))
        
        crop = image.crop((x1_p, y1_p, x2_p, y2_p))
        crop.save(f"debug_crops/candidate_{i}.png")
        
        inputs = processor_siglip(text=candidate_labels, images=crop, return_tensors="pt", padding=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = model_siglip(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
        siglip_score = probs[0] # Probability of "yellow jersey"
        
        if siglip_score > threshold:
            print(f"  [Match] Object {i} SigLIP Score: {siglip_score:.3f}")
            verified_indices.append(i)
        else:
            print(f"  [Reject] Object {i} SigLIP Score: {siglip_score:.3f}")
            
    return verified_indices

def refine_masks_sam2(image, boxes, processor_sam2, model_sam2):
    """Run SAM 2 with robust mask selection."""
    width, height = image.size
    # SAM 2 expects a list of lists for boxes
    inputs_sam2 = processor_sam2(image, input_boxes=[boxes.tolist()], return_tensors="pt")
    inputs_sam2_gpu = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs_sam2.items()}
    
    with torch.no_grad():
        outputs_sam2 = model_sam2(**inputs_sam2_gpu, multimask_output=True)
    
    final_masks = []
    for i in range(len(boxes)):
        iou_scores = outputs_sam2.iou_scores[0, i].cpu().numpy()
        # Prefer level 2 (whole) unless it's way worse than level 0 or 1
        best_idx = 2 if iou_scores[2] > (np.max(iou_scores) - 0.15) else np.argmax(iou_scores)
        raw_mask = outputs_sam2.pred_masks[0, i, best_idx]
        mask_full = torch.nn.functional.interpolate(raw_mask.unsqueeze(0).unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0, 0]
        final_masks.append((mask_full > 0.0).cpu().numpy().astype(bool))
        
    return np.array(final_masks)

def main():
    image = Image.open(IMAGE_PATH).convert("RGB")
    p_dino, m_dino, p_sam2, m_sam2, p_siglip, m_siglip = load_models()
    
    # STAGE 1: Discovery with Grounding DINO
    # More specific discovery to avoid crowd as much as possible
    discovery_prompt = "basketball player in game . " 
    inputs_dino = p_dino(images=image, text=discovery_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs_dino = m_dino(**inputs_dino)
    
    # Lower threshold to ensure we catch the player on the far right
    results_dino = p_dino.post_process_grounded_object_detection(
        outputs_dino, inputs_dino.input_ids, threshold=0.1, text_threshold=0.1,
        target_sizes=torch.Tensor([[image.height, image.width]]).to(DEVICE)
    )[0]
    
    boxes = results_dino["boxes"].cpu().numpy()
    scores = results_dino["scores"].cpu().numpy()
    
    if len(boxes) == 0:
        print("No players found at all.")
        return

    # Combine for NMS
    predictions = np.concatenate([boxes, scores.reshape(-1, 1)], axis=1)
    # More aggressive NMS (0.3 instead of 0.5) to merge overlapping detections of the same person
    mask_to_keep = sv.box_non_max_suppression(predictions, iou_threshold=0.3)
    boxes = boxes[mask_to_keep]
    
    # NEW: Filter out very large boxes (likely background/crowd chunks)
    # If a box is more than 20% of the image area, it's probably not a single player
    img_area = image.width * image.height
    filtered_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        if box_area < (img_area * 0.2): 
            filtered_boxes.append(box)
    
    boxes = np.array(filtered_boxes)
    if len(boxes) == 0:
        print("All boxes filtered out by size.")
        return
        
    # STAGE 2: Verification with SigLIP (The Google Model)
    # Stricter threshold and better labels
    target_verification = "basketball player in yellow jersey"
    verified_indices = verify_with_google_siglip(image, boxes, target_verification, p_siglip, m_siglip, threshold=0.75)
    
    if len(verified_indices) > 0:
        verified_boxes = boxes[verified_indices]
        
        # STAGE 3: Masking with SAM 2
        print(f"Refining masks for {len(verified_boxes)} verified players...")
        masks = refine_masks_sam2(image, verified_boxes, p_sam2, m_sam2)
        
        # STAGE 4: Visualization
        detections = sv.Detections(
            xyxy=verified_boxes, 
            mask=masks,
            class_id=np.arange(len(verified_boxes))
        )
        annotated = np.array(image)
        # Higher contrast visualization
        annotated = sv.MaskAnnotator(opacity=0.6).annotate(scene=annotated, detections=detections)
        annotated = sv.BoxAnnotator(thickness=3).annotate(scene=annotated, detections=detections)
        annotated = sv.LabelAnnotator(text_scale=0.8).annotate(scene=annotated, detections=detections, labels=[f"Yellow Player {i}" for i in range(len(verified_boxes))])
        
        cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"SUCCESS: Saved {OUTPUT_PATH}. Found {len(verified_boxes)} yellow shirt players.")
    else:
        print("SigLIP rejected all candidates. Try adjusting thresholds.")

if __name__ == "__main__":
    main()
