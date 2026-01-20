"""
Production Inference Script for Connector Inspection
Single YOLOv11 model - detects location AND classifies quality
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import time

class ConnectorInspector:
    """Production-ready connector inspection system."""
    
    def __init__(self, model_path, conf_threshold=0.25, device='0'):
        """
        Initialize inspector.
        
        Args:
            model_path: Path to trained YOLOv11 model (.pt file)
            conf_threshold: Confidence threshold (0.25 default)
            device: '0' for GPU, 'cpu' for CPU
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Class names
        self.class_names = {
            0: 'connector_ok',
            1: 'connector_not_ok'
        }
        
        # Colors for visualization (BGR)
        self.colors = {
            0: (0, 255, 0),   # Green for OK
            1: (0, 0, 255)    # Red for NOT_OK
        }
        
        print(f"✅ Connector Inspector initialized")
        print(f"   Model: {model_path}")
        print(f"   Confidence threshold: {conf_threshold}")
        print(f"   Device: {device}")
    
    def inspect(self, image_path):
        """
        Inspect a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict with inspection results
        """
        # Run inference
        results = self.model(image_path, conf=self.conf_threshold, device=self.device, verbose=False)
        
        result = results[0]
        boxes = result.boxes
        
        # Parse results
        detections = []
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            
            detections.append({
                'class_id': class_id,
                'class_name': self.class_names[class_id],
                'confidence': confidence,
                'bbox': bbox.tolist(),
                'status': 'PASS' if class_id == 0 else 'FAIL'
            })
        
        return {
            'image_path': str(image_path),
            'num_detections': len(detections),
            'detections': detections,
            'overall_status': 'PASS' if all(d['status'] == 'PASS' for d in detections) else 'FAIL'
        }
    
    def inspect_and_visualize(self, image_path, output_path=None):
        """
        Inspect image and create annotated visualization.
        
        Args:
            image_path: Path to image
            output_path: Where to save annotated image (optional)
            
        Returns:
            Inspection results dict
        """
        # Get inspection results
        inspection = self.inspect(image_path)
        
        # Load image for visualization
        img = cv2.imread(str(image_path))
        
        # Draw detections
        for det in inspection['detections']:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            color = self.colors[det['class_id']]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{class_name.replace('connector_', '').upper()}: {confidence:.2%}"
            
            # Label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Label text
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        
        # Add overall status
        status_color = (0, 255, 0) if inspection['overall_status'] == 'PASS' else (0, 0, 255)
        status_text = f"OVERALL: {inspection['overall_status']}"
        cv2.putText(img, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.2, status_color, 3)
        
        # Save or display
        if output_path:
            cv2.imwrite(str(output_path), img)
            print(f"💾 Saved annotated image to: {output_path}")
        
        return inspection, img
    
    def batch_inspect(self, image_folder, output_folder=None, save_results=True):
        """
        Inspect a batch of images.
        
        Args:
            image_folder: Folder containing images
            output_folder: Where to save annotated images (optional)
            save_results: Save results to JSON file
            
        Returns:
            List of inspection results
        """
        image_folder = Path(image_folder)
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        images = []
        for ext in image_extensions:
            images.extend(list(image_folder.glob(ext)))
        
        print(f"\n🔍 Inspecting {len(images)} images...")
        
        results = []
        start_time = time.time()
        
        for i, img_path in enumerate(images, 1):
            # Inspect
            inspection = self.inspect(img_path)
            results.append(inspection)
            
            # Save annotated image if requested
            if output_folder:
                output_folder = Path(output_folder)
                output_folder.mkdir(exist_ok=True)
                
                output_path = output_folder / f"{img_path.stem}_annotated.jpg"
                _, annotated_img = self.inspect_and_visualize(img_path, output_path)
            
            # Progress
            if i % 10 == 0 or i == len(images):
                print(f"   Progress: {i}/{len(images)} images processed")
        
        elapsed = time.time() - start_time
        fps = len(images) / elapsed
        
        print(f"\n✅ Batch inspection complete!")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Speed: {fps:.2f} FPS")
        
        # Summary statistics
        pass_count = sum(1 for r in results if r['overall_status'] == 'PASS')
        fail_count = len(results) - pass_count
        
        print(f"\n📊 Summary:")
        print(f"   Total: {len(results)}")
        print(f"   PASS: {pass_count} ({pass_count/len(results)*100:.1f}%)")
        print(f"   FAIL: {fail_count} ({fail_count/len(results)*100:.1f}%)")
        
        # Save results to JSON
        if save_results:
            import json
            results_path = Path(image_folder) / 'inspection_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {results_path}")
        
        return results