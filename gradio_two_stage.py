"""
Modern Two-Stage Alternator Connector Inspector
YOLO Detection + EfficientNet Classification
"""
import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO
import argparse

# Global variables for models
yolo_model = None
classifier_model = None
processor = None

def load_models(yolo_path, classifier_path):
    """Load YOLO and classifier models"""
    global yolo_model, classifier_model, processor
    
    print("Loading YOLO detector...")
    yolo_model = YOLO(yolo_path)
    
    print("Loading classifier...")
    processor = AutoImageProcessor.from_pretrained(classifier_path)
    classifier_model = AutoModelForImageClassification.from_pretrained(classifier_path)
    classifier_model.eval()
    
    print("✓ Models loaded successfully!")

def detect_and_classify(image):
    """
    Two-stage pipeline:
    1. YOLO detects connector
    2. Classifier determines OK/NOT_OK
    """
    if image is None:
        return None, "Please upload an image", {}
    
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Stage 1: YOLO Detection
    results = yolo_model(pil_image, conf=0.25, verbose=False)
    
    if len(results[0].boxes) == 0:
        return pil_image, "⚠️ No connector detected", {
            "Status": "No Detection",
            "Confidence": "N/A"
        }
    
    # Get best detection
    boxes = results[0].boxes
    best_box = boxes[0]  # Highest confidence
    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
    detection_conf = float(best_box.conf[0])
    
    # Crop connector region (with padding)
    img_array = np.array(pil_image)
    h, w = img_array.shape[:2]
    
    # Add 10% padding
    padding = 0.1
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)
    
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    
    cropped = img_array[y1:y2, x1:x2]
    cropped_pil = Image.fromarray(cropped)
    
    # Stage 2: Classification
    inputs = processor(images=cropped_pil, return_tensors="pt")
    
    with torch.no_grad():
        outputs = classifier_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = probabilities.argmax().item()
        confidence = probabilities[0][predicted_class].item() * 100
    
    # Map to labels
    labels = ["NOT_OK", "OK"]
    prediction = labels[predicted_class]
    
    # Create visualization
    vis_image = pil_image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # Draw bounding box
    if prediction == "OK":
        color = "green"
        status_emoji = "✅"
    else:
        color = "red"
        status_emoji = "❌"
    
    # Draw box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
    
    # Draw label background
    label_text = f"{prediction} ({confidence:.1f}%)"
    
    # Use default font
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Get text size
    bbox = draw.textbbox((0, 0), label_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Draw label background
    label_bg = [x1, y1 - text_height - 10, x1 + text_width + 10, y1]
    draw.rectangle(label_bg, fill=color)
    
    # Draw text
    draw.text((x1 + 5, y1 - text_height - 5), label_text, fill="white", font=font)
    
    # Result message
    if prediction == "OK":
        message = f"{status_emoji} **Connector OK** - Properly connected"
        description = "The alternator connector is securely attached and properly seated."
    else:
        message = f"{status_emoji} **Connector NOT OK** - Issue detected"
        description = "⚠️ The connector appears disconnected or improperly seated. Please check the connection."
    
    # Detailed info
    info = {
        "🎯 Final Result": f"{prediction}",
        "📊 Classification Confidence": f"{confidence:.2f}%",
        "🔍 Detection Confidence": f"{detection_conf * 100:.2f}%",
        "📦 Bounding Box": f"({x1}, {y1}) → ({x2}, {y2})",
        "📝 Description": description
    }
    
    return vis_image, message, info

def create_interface():
    """Create modern Gradio interface"""
    
    with gr.Blocks(title="Alternator Connector Inspector") as demo:
        
        # Header
        gr.Markdown("""
        # 🔧 Alternator Connector Quality Inspector
        ### Two-Stage AI Detection System (YOLO + EfficientNet)
        Upload an image to automatically detect and classify alternator connector status.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📤 Upload Image")
                input_image = gr.Image(
                    type="pil",
                    label="Alternator Image",
                    height=400
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    classify_btn = gr.Button("🔍 Inspect Connector", variant="primary", scale=2)
                
                # Examples
                gr.Markdown("### 📋 Sample Images")
                gr.Examples(
                    examples=[
                        ["data/ok/example1.jpg"],
                        ["data/not_ok/example1.jpg"],
                    ],
                    inputs=input_image,
                    label="Try these examples"
                )
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 📊 Inspection Results")
                output_image = gr.Image(
                    type="pil",
                    label="Detection & Classification",
                    height=400
                )
                
                result_text = gr.Markdown(
                    value="*Upload an image to begin inspection*",
                    label="Status"
                )
                
                detailed_info = gr.JSON(
                    label="Detailed Analysis",
                    visible=True
                )
        
        # Info section
        gr.Markdown("""
        ---
        ### 🎯 How it works:
        1. **YOLO Detection** - Locates the connector in the image
        2. **Smart Cropping** - Extracts connector region with padding
        3. **EfficientNet Classification** - Determines if connection is OK or NOT_OK
        
        ### 📈 Performance:
        - **Detection Accuracy**: 95%+ (YOLO mAP@0.5)
        - **Classification Accuracy**: 93.55% (Two-stage pipeline)
        - **Previous Single-Stage**: 74.19%
        - **Improvement**: +19.36%
        
        ### ✅ Status Indicators:
        - 🟢 **Green Box** = Connector OK
        - 🔴 **Red Box** = Connector NOT OK
        """)
        
        # Event handlers
        classify_btn.click(
            fn=detect_and_classify,
            inputs=input_image,
            outputs=[output_image, result_text, detailed_info]
        )
        
        clear_btn.click(
            fn=lambda: (None, "*Upload an image to begin inspection*", {}),
            inputs=None,
            outputs=[output_image, result_text, detailed_info]
        )
        
        input_image.change(
            fn=detect_and_classify,
            inputs=input_image,
            outputs=[output_image, result_text, detailed_info]
        )
    
    return demo

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Connector Inspector")
    parser.add_argument('--yolo-model', type=str, 
                       default='connector_detector/yolov8_connector/weights/best.pt',
                       help='Path to YOLO model')
    parser.add_argument('--classifier-model', type=str,
                       default='best_model_hf',
                       help='Path to classifier model')
    parser.add_argument('--share', action='store_true',
                       help='Create public link')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port number')
    
    args = parser.parse_args()
    
    # Load models
    load_models(args.yolo_model, args.classifier_model)
    
    # Create and launch interface
    demo = create_interface()
    
    print("\n" + "="*50)
    print("🚀 Launching Alternator Connector Inspector...")
    print("="*50)
    
    demo.launch(
        share=True,
        server_port=args.port,
        server_name="0.0.0.0"
    )

if __name__ == "__main__":
    main()