import torch
from PIL import Image
import numpy as np
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Model, Sam2Processor
import supervision as sv

# --- USER CONFIGURATION ---
# PLEASE UPDATE THIS PATH TO YOUR WHEEL IMAGE
IMAGE_PATH = r"D:\Model_Training\basket.jpg"  # Example path, change to your actual wheel image
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
    print(f"Error loading image at {IMAGE_PATH}: {e}")
    print("Please update IMAGE_PATH in the script to point to your wheel image.")
    exit()

# 3. Grounding DINO: Find the "center cap" box
# "center cap" is more specific than "wheel cap"
text_prompt = "basketball player with white shirt." 
print(f"Running Grounding DINO with prompt: '{text_prompt}'...")

inputs_dino = processor_dino(images=image, text=text_prompt, return_tensors="pt").to(device)
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
    # 4. SAM 2: Use Point Prompting (Center of Box)
    # Instead of giving the whole box (which includes reflections), we give the CENTER point.
    # This forces SAM 2 to start at the cap and grow outwards, avoiding the shiny spokes.
    
    print("Running SAM 2 with POINT PROMPTS (to avoid reflections)...")
    
    # Calculate center points for all boxes
    boxes = results_dino["boxes"].cpu().numpy()
    centers = []
    for box in boxes:
        cx = float((box[0] + box[2]) / 2)
        cy = float((box[1] + box[3]) / 2)
        centers.append([cx, cy])
    
    # Prepare inputs for SAM 2
    # input_points shape: (batch_size, num_objects, num_points, 2)
    # input_labels shape: (batch_size, num_objects, num_points) -> 1 = foreground point
    
    # We have 1 image (batch_size=1)
    # centers list is (num_objects, 2)
    # We want: [ [ [x, y] ] ] for each object
    
    # Nesting Level 1: Image (Batch)
    # Nesting Level 2: Objects
    # Nesting Level 3: Points (we have 1 point per object)
    # Nesting Level 4: Coords
    
    formatted_points = []
    formatted_labels = []
    
    # For the single image in the batch:
    image_points = []
    image_labels = []
    
    for c in centers:
        image_points.append([c]) # Object level -> Point level
        image_labels.append([1]) # Object level -> Point level
        
    formatted_points.append(image_points) # Image level
    formatted_labels.append(image_labels) # Image level
    
    inputs_sam2 = processor_sam2(
        image, 
        input_points=formatted_points, 
        input_labels=formatted_labels, 
        return_tensors="pt"
    )
    
    # Move to GPU
    inputs_sam2_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_sam2.items()}
    
    with torch.no_grad():
        # multimask_output=True allows SAM 2 to resolve ambiguity (small vs large)
        outputs_sam2 = model_sam2(**inputs_sam2_gpu, multimask_output=True)
    
    # Process masks and select Level 2 (the best for this case)
    print("Selecting Level 2 masks and applying morphological cleanup...")
    cleaned_masks = []
    
    for i in range(len(centers)):
        # Select mask Level 2 (as determined from our visual test)
        raw_mask = outputs_sam2.pred_masks[0, i, 2]
        
        # Upsample
        mask_full = torch.nn.functional.interpolate(
            raw_mask.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )[0, 0]
        
        # Threshold to binary
        binary_mask = (mask_full > 0.0).cpu().numpy().astype(np.uint8)
        
        # Morphological Cleanup: Remove noise and fill small holes
        # Opening removes small speckles outside
        # Closing fills small holes inside
        kernel = np.ones((5,5), np.uint8)
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        cleaned_masks.append(cleaned_mask.astype(bool))
    
    # Use supervision to visualize the final result
    detections = sv.Detections(
        xyxy=boxes,
        mask=np.array(cleaned_masks),
        class_id=np.arange(len(boxes)),
        tracker_id=None
    )

    mask_annotator = sv.MaskAnnotator(color=sv.Color.from_hex("#00FF00")) # Pure green
    box_annotator = sv.BoxAnnotator()
    
    annotated_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
    # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections) # Uncomment if you want boxes
    
    cv2.imwrite("final_basketball_player_result.png", annotated_image)
    print("Successfully saved 'final_basketball_player_result.png'!")

else:
    print("No objects found. Try adjusting the prompt or threshold.")
