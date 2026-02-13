from ultralytics import YOLO
import cv2
import sys
import numpy as np

# ============================================ 
# DIAGNOSTIC TOOL
# ============================================ 
# Usage: python verify_model_confidence.py <image_path> <model_path>

def analyze_image(image_path, model_path):
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Reading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    # Run inference with low threshold to see ALL candidates
    results = model(img, conf=0.10, verbose=False)
    
    print(f"\n{'='*40}")
    print(f"DETECTION REPORT")
    print(f"{'='*40}")
    
    if len(results[0].boxes) == 0:
        print("No detections found even at 0.10 confidence.")
        return

    # box.cls map
    names = results[0].names
    
    for i, box in enumerate(results[0].boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        print(f"\nBox {i+1}: {names[cls_id]} (Confidence: {conf:.4f})")
        print(f"   Coords: [{x1}, {y1}, {x2}, {y2}]")
        
        # Check specific color confusion if available
        # This requires the model to return all class probs, which standard predict() might not do 
        # unless 'save_conf=True' or accessing raw outputs. 
        # For now, we trust the top-1 confidence is low if there is confusion. 
        
        if conf < 0.7:
             print("   ⚠️  LOW CONFIDENCE! Model is uncertain.")
             print("      Possible causes: Bad lighting, blur, or confusion with similar class.")
        else:
             print("   ✅ High Confidence.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_model_confidence.py <image_path> <model_path>")
        # Default for testing
        # analyze_image("test.jpg", "yolo11m.pt")
    else:
        analyze_image(sys.argv[1], sys.argv[2])
