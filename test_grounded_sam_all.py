import torch
from PIL import Image
import numpy as np
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Model, Sam2Processor
import supervision as sv

# Paths to your local models
DINO_PATH = r"datavision_hf_models\grounding-dino-base"
SAM2_PATH = r"datavision_hf_models\sam2-hiera-large"
IMAGE_PATH = r"D:\Model_Training\traffic.jpg"

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
text_prompt = "traffic light . "
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
    # 5. Run SAM 2 for all detected boxes
    print("Running SAM 2 for all masks...")
    boxes = results_dino["boxes"].cpu().numpy()
    labels = results_dino["labels"]
    
    inputs_sam2 = processor_sam2(image, input_boxes=[boxes], return_tensors="pt")
    inputs_sam2_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_sam2.items()}
    
    with torch.no_grad():
        # multimask_output=True gives 3 levels of masks (part, sub-part, whole)
        outputs_sam2 = model_sam2(**inputs_sam2_gpu, multimask_output=True)
    
    # Process masks: pick the level with the highest IoU for each object
    final_masks = []
    for i in range(len(boxes)):
        # Select the best mask level based on IoU score
        best_mask_idx = torch.argmax(outputs_sam2.iou_scores[0, i]).item()
        raw_mask = outputs_sam2.pred_masks[0, i, best_mask_idx]
        
        # Upsample mask to original image size
        mask_full = torch.nn.functional.interpolate(
            raw_mask.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )[0, 0]
        
        binary_mask = (mask_full > 0.0).cpu().numpy().astype(bool)
        final_masks.append(binary_mask)
    
    final_masks = np.array(final_masks)
    
    # 6. Visualize all detections using Supervision
    print("Annotating image with all detections...")
    detections = sv.Detections(
        xyxy=boxes,
        mask=final_masks,
        class_id=np.arange(len(boxes)) # Assigning unique IDs for visualization
    )
    
    annotated_image = np.array(image)
    
    # Use standard annotators
    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    
    # Create labels with the original class names from DINO
    labels_display = [f"{label} {i}" for i, label in enumerate(labels)]
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels_display)
    
    # Save the final result
    output_filename = "test_result_all_players.png"
    cv2.imwrite(output_filename, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print(f"Successfully saved {output_filename} with all {len(boxes)} detections.")

else:
    print("No objects detected.")
