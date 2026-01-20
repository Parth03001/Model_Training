"""
Two-Stage Inference Pipeline:
Stage 1: Detect connector with YOLO
Stage 2: Classify connection state with ViT
"""
import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np

class TwoStageConnectorInspector:
    def __init__(
        self, 
        yolo_model_path='connector_detector/yolov8_connector/weights/best.pt',
        classifier_model_path='best_model_hf',
        conf_threshold=0.25
    ):
        """
        Initialize two-stage pipeline
        
        Args:
            yolo_model_path: Path to YOLO detector weights
            classifier_model_path: Path to ViT classifier
            conf_threshold: Detection confidence threshold
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Stage 1: YOLO Detector
        print("Loading YOLO detector...")
        self.detector = YOLO(yolo_model_path)
        self.conf_threshold = conf_threshold
        
        # Stage 2: ViT Classifier
        print("Loading ViT classifier...")
        self.processor = AutoImageProcessor.from_pretrained(classifier_model_path)
        self.classifier = AutoModelForImageClassification.from_pretrained(classifier_model_path)
        self.classifier.to(self.device)
        self.classifier.eval()
        
        self.labels = ['OK', 'NOT_OK']
        print("✓ Two-stage pipeline loaded successfully!")
    
    def detect_connector(self, image):
        """
        Stage 1: Detect connector region
        
        Returns:
            cropped_image: PIL Image of connector region
            detection_conf: Detection confidence
            bbox: Bounding box coordinates (x1, y1, x2, y2)
        """
        # Run detection
        results = self.detector(image, conf=self.conf_threshold, verbose=False)
        
        # Check if connector detected
        if len(results[0].boxes) == 0:
            return None, 0.0, None
        
        # Get first (highest confidence) detection
        box = results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        detection_conf = float(box.conf[0])
        
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Add padding (10% on each side)
        width = x2 - x1
        height = y2 - y1
        padding_w = width * 0.1
        padding_h = height * 0.1
        
        x1 = max(0, x1 - padding_w)
        y1 = max(0, y1 - padding_h)
        x2 = min(image.width, x2 + padding_w)
        y2 = min(image.height, y2 + padding_h)
        
        # Crop connector region
        cropped = image.crop((int(x1), int(y1), int(x2), int(y2)))
        
        return cropped, detection_conf, (x1, y1, x2, y2)
    
    def classify_connector(self, cropped_image):
        """
        Stage 2: Classify connection state
        
        Returns:
            result: 'OK' or 'NOT_OK'
            confidence: Classification confidence
            probabilities: Dict of class probabilities
        """
        # Process image
        inputs = self.processor(images=cropped_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.classifier(pixel_values=pixel_values)
            probs = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
        
        # Get probabilities
        prob_dict = {
            self.labels[i]: float(probs[0][i]) 
            for i in range(len(self.labels))
        }
        
        result = self.labels[predicted_class]
        confidence = prob_dict[result]
        
        return result, confidence, prob_dict
    
    def inspect(self, image):
        """
        Full two-stage inspection pipeline
        
        Args:
            image: PIL Image, numpy array, or path to image
            
        Returns:
            dict with:
                - detection_success: bool
                - detection_confidence: float
                - classification_result: str ('OK' or 'NOT_OK')
                - classification_confidence: float
                - probabilities: dict
                - cropped_connector: PIL Image (if detected)
                - bbox: tuple (x1, y1, x2, y2) if detected
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Stage 1: Detect
        cropped, det_conf, bbox = self.detect_connector(image)
        
        if cropped is None:
            return {
                'detection_success': False,
                'detection_confidence': 0.0,
                'classification_result': 'NO_DETECTION',
                'classification_confidence': 0.0,
                'probabilities': {},
                'cropped_connector': None,
                'bbox': None,
                'error': 'Connector not detected in image'
            }
        
        # Stage 2: Classify
        result, cls_conf, probs = self.classify_connector(cropped)
        
        return {
            'detection_success': True,
            'detection_confidence': det_conf,
            'classification_result': result,
            'classification_confidence': cls_conf,
            'probabilities': probs,
            'cropped_connector': cropped,
            'bbox': bbox
        }
    
    def format_result(self, inspection_result):
        """Format inspection result for display"""
        if not inspection_result['detection_success']:
            return f"❌ {inspection_result['error']}"
        
        result = inspection_result['classification_result']
        conf = inspection_result['classification_confidence']
        det_conf = inspection_result['detection_confidence']
        
        if result == 'OK':
            status = f"✅ PASS - Connector OK"
        else:
            status = f"❌ FAIL - Connector NOT OK"
        
        return (
            f"{status}\n"
            f"Detection Confidence: {det_conf:.1%}\n"
            f"Classification Confidence: {conf:.1%}"
        )

# Example usage
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    
    if len(sys.argv) < 2:
        print("Usage: python two_stage_inference.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Initialize inspector
    inspector = TwoStageConnectorInspector()
    
    # Inspect
    result = inspector.inspect(image_path)
    
    # Display results
    print("\n" + "="*50)
    print(inspector.format_result(result))
    print("="*50)
    
    if result['detection_success']:
        print(f"\nDetailed Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.2%}")
        
        # Show original and cropped
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original
        original = Image.open(image_path)
        axes[0].imshow(original)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Draw bbox
        if result['bbox']:
            from matplotlib.patches import Rectangle
            x1, y1, x2, y2 = result['bbox']
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor='lime', facecolor='none')
            axes[0].add_patch(rect)
        
        # Cropped
        axes[1].imshow(result['cropped_connector'])
        axes[1].set_title(f"Detected Connector\n{result['classification_result']}")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('inspection_result.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to inspection_result.png")