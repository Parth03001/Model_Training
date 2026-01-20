"""
Stage 1: Train YOLO Connector Detector
"""
from ultralytics import YOLO
import os

def train_yolo_detector(data_yaml_path, epochs=50, img_size=640):
    """
    Train YOLOv8 to detect connectors
    
    Args:
        data_yaml_path: Path to data.yaml file (from annotation tool)
        epochs: Training epochs
        img_size: Image size
    """
    
    # Load YOLOv8 nano model (fast and accurate for single object)
    model = YOLO('yolov8n.pt')  # nano model
    # Alternative: YOLO('yolov8s.pt') for more accuracy
    
    print("🚀 Training YOLO Connector Detector...")
    
    # Train
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        patience=10,
        save=True,
        project='connector_detector',
        name='yolov8_connector',
        exist_ok=True,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )
    
    print(f"\n✓ Training complete!")
    print(f"Best model saved to: connector_detector/yolov8_connector/weights/best.pt")
    
    # Validate
    metrics = model.val()
    print(f"\nValidation mAP@0.5: {metrics.box.map50:.3f}")
    print(f"Validation mAP@0.5:0.95: {metrics.box.map:.3f}")
    
    return model

def create_data_yaml(train_img_dir, val_img_dir, output_path='data.yaml'):
    """
    Create data.yaml if not using annotation tool export
    """
    yaml_content = f"""
train: {train_img_dir}
val: {val_img_dir}
nc: 1
names: ['connector']
"""
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Created {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    
    args = parser.parse_args()
    
    train_yolo_detector(args.data, args.epochs, args.img_size)