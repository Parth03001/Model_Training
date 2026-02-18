
import torch
from PIL import Image
import requests
import numpy as np
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Model, Sam2Processor
import supervision as sv

# Paths to your local models
DINO_PATH = r"datavision_hf_models\grounding-dino-base"
SAM2_PATH = r"datavision_hf_models\sam2-hiera-large"
IMAGE_PATH = r"D:\Model_Training\basket.jpg"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load Grounding DINO
print("Loading Grounding DINO...")
processor_dino = AutoProcessor.from_pretrained(DINO_PATH)
model_dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_PATH).to(device)

# 2. Load SAM 2
print("Loading SAM 2...")
processor_sam2 = Sam2Processor.from_pretrained(SAM2_PATH)
model_sam2 = Sam2Model.from_pretrained(SAM2_PATH).to(device)

# 3. Load and prepare image
image = Image.open(IMAGE_PATH).convert("RGB")
text_prompt = "helmet . "
inputs_dino = processor_dino(images=image, text=text_prompt, return_tensors="pt").to(device)

# 4. Run Grounding DINO
print(f"Running Grounding DINO with prompt: '{text_prompt}'...")
with torch.no_grad():
    outputs_dino = model_dino(**inputs_dino)

# Post-process DINO results
width, height = image.size
target_sizes = torch.Tensor([[height, width]]).to(device)
results_dino = processor_dino.post_process_grounded_object_detection(
    outputs_dino,
    inputs_dino.input_ids,
    threshold=0.2, 
    text_threshold=0.2,
    target_sizes=target_sizes
)[0]

print(f"Found {len(results_dino['boxes'])} objects.")

if len(results_dino["boxes"]) > 0:
    # 5. Run SAM 2
    print("Running SAM 2 for masks...")
    boxes = results_dino["boxes"].cpu().numpy()
    labels = results_dino["labels"] # Get the text labels
    
    inputs_sam2 = processor_sam2(image, input_boxes=[boxes], return_tensors="pt")
    # Only move tensors to GPU
    inputs_sam2_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_sam2.items()}
    
    with torch.no_grad():
        # Enable multimask output to get different levels of granularity (whole vs part)
        outputs_sam2 = model_sam2(**inputs_sam2_gpu, multimask_output=True)
    
    # Process each mask individually to clean it up
    # We will visualize ALL 3 mask levels for the first box to understand what SAM 2 is seeing
    
    # Get the first object's results (assuming only 1 dog/tail found)
    box_idx = 0 
    box = boxes[box_idx]
    label = labels[box_idx]
    
    # Create a canvas to show all 3 masks side-by-side
    canvas = np.zeros((height, width * 3, 3), dtype=np.uint8)
    
    # Original image for background
    original_img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    print(f"Analyzing 3 mask levels for object: {label}...")
    
    for mask_idx in range(3):
        # Get raw mask for this level
        # outputs_sam2.pred_masks shape: (batch, num_objects, 3_levels, h, w)
        raw_mask = outputs_sam2.pred_masks[0, box_idx, mask_idx]
        iou_score = outputs_sam2.iou_scores[0, box_idx, mask_idx].item()
        
        # Upsample
        mask_full = torch.nn.functional.interpolate(
            raw_mask.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )[0, 0]
        
        # Threshold
        binary_mask = (mask_full > 0.0).cpu().numpy().astype(np.uint8)
        
        # Create a colored overlay
        colored_mask = np.zeros_like(original_img_cv)
        colored_mask[binary_mask == 1] = [0, 255, 0] # Green for mask
        
        # Blend
        overlayed_img = cv2.addWeighted(original_img_cv, 0.7, colored_mask, 0.3, 0)
        
        # Draw the box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlayed_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Add Label text
        text = f"Level {mask_idx} (IoU: {iou_score:.2f})"
        cv2.putText(overlayed_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Place in canvas
        canvas[:, mask_idx * width : (mask_idx + 1) * width] = overlayed_img
        
        print(f"  Level {mask_idx}: IoU {iou_score:.4f}")

    output_filename = "test_result_levels.png"
    cv2.imwrite(output_filename, canvas)
    print(f"Successfully saved {output_filename} - Check this to see if the tail is in Level 0 or 1.")

else:
    print("No objects detected.")
