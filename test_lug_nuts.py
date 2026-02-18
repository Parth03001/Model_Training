import torch
from PIL import Image
import numpy as np
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Model, Sam2Processor
import supervision as sv

# --- USER CONFIGURATION ---
IMAGE_PATH = r"D:\Model_Training\datavision-platform\image.png"
TEXT_PROMPT = "lug nut." # <--- CHANGED PROMPT HERE
# --------------------------

# Paths to models
DINO_PATH = r"datavision_hf_models\grounding-dino-base"
SAM2_PATH = r"datavision_hf_models\sam2-hiera-large"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load Models
print("Loading Grounding DINO...")
processor_dino = AutoProcessor.from_pretrained(DINO_PATH)
model_dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_PATH).to(device)

print("Loading SAM 2...")
processor_sam2 = Sam2Processor.from_pretrained(SAM2_PATH)
model_sam2 = Sam2Model.from_pretrained(SAM2_PATH).to(device)

# 2. Load Image
try:
    image = Image.open(IMAGE_PATH).convert("RGB")
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# 3. Grounding DINO
print(f"Running Grounding DINO with prompt: '{TEXT_PROMPT}'...")

inputs_dino = processor_dino(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
with torch.no_grad():
    outputs_dino = model_dino(**inputs_dino)

# Post-process DINO
width, height = image.size
target_sizes = torch.Tensor([[height, width]]).to(device)
results_dino = processor_dino.post_process_grounded_object_detection(
    outputs_dino,
    inputs_dino.input_ids,
    threshold=0.25,
    text_threshold=0.25,
    target_sizes=target_sizes
)[0]

print(f"Found {len(results_dino['boxes'])} objects.")

if len(results_dino["boxes"]) > 0:
    # 4. SAM 2 with Point Prompts
    print("Running SAM 2 with POINT PROMPTS...")
    
    # Calculate centers
    boxes = results_dino["boxes"].cpu().numpy()
    centers = []
    for box in boxes:
        cx = float((box[0] + box[2]) / 2)
        cy = float((box[1] + box[3]) / 2)
        centers.append([cx, cy])
    
    # Prepare inputs (Nest Level 4 structure)
    formatted_points = []
    formatted_labels = []
    image_points = []
    image_labels = []
    
    for c in centers:
        image_points.append([c])
        image_labels.append([1])
        
    formatted_points.append(image_points)
    formatted_labels.append(image_labels)
    
    inputs_sam2 = processor_sam2(
        image, 
        input_points=formatted_points, 
        input_labels=formatted_labels, 
        return_tensors="pt"
    )
    
    inputs_sam2_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_sam2.items()}
    
    with torch.no_grad():
        outputs_sam2 = model_sam2(**inputs_sam2_gpu, multimask_output=True)
    
    # Visualization
    print("Visualizing results...")
    
    # We will assume Level 1 (middle) is good for nuts, but let's select best score just in case
    # Or stick to Level 2 (Whole Object) if they are simple shapes. 
    # For small nuts, Level 0 or 1 might be better than 2 (which might bleed).
    # Let's try Level 1 as a safe default for small parts.
    
    cleaned_masks = []
    for i in range(len(centers)):
        # Try Level 1 for nuts (often tighter)
        raw_mask = outputs_sam2.pred_masks[0, i, 1] 
        
        mask_full = torch.nn.functional.interpolate(
            raw_mask.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )[0, 0]
        
        binary_mask = (mask_full > 0.0).cpu().numpy().astype(np.uint8)
        cleaned_masks.append(binary_mask.astype(bool))
    
    detections = sv.Detections(
        xyxy=boxes,
        mask=np.array(cleaned_masks),
        class_id=np.arange(len(boxes)),
        tracker_id=None
    )

    mask_annotator = sv.MaskAnnotator(color=sv.Color.from_hex("#FF0000")) # Red for nuts
    box_annotator = sv.BoxAnnotator()
    
    annotated_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    
    cv2.imwrite("lug_nuts_result.png", annotated_image)
    print("Successfully saved 'lug_nuts_result.png'!")

else:
    print("No objects found.")
