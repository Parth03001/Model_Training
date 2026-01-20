"""
YOLOv11 Single-Stage Training for Connector Inspection
Train one model that does both detection AND classification
"""

from ultralytics import YOLO
import os
from pathlib import Path

def train_yolo11_single_stage(data_yaml_path, epochs=100, img_size=640, 
                               model_size='m', device=0):
    """
    Train YOLOv11 for single-stage connector inspection
    
    Args:
        data_yaml_path: Path to data.yaml file
        epochs: Training epochs (100 recommended for 1000 images/class)
        img_size: Image size (640 standard)
        model_size: 'n'=nano, 's'=small, 'm'=medium, 'l'=large, 'x'=xlarge
        device: GPU device (0 for first GPU, 'cpu' for CPU)
    """
    
    # Model selection guide
    model_guide = {
        'n': 'Nano - fastest, lowest accuracy (~2.6M params)',
        's': 'Small - good balance for edge devices (~9M params)',
        'm': 'Medium - RECOMMENDED for your use case (~20M params)',
        'l': 'Large - maximum accuracy (~25M params)',
        'x': 'X-Large - overkill for most cases (~56M params)'
    }
    
    print(f"\n🚀 Training YOLOv11-{model_size.upper()}")
    print(f"   {model_guide[model_size]}")
    print(f"   Dataset: {data_yaml_path}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {img_size}")
    print(f"   Device: {device}\n")
    
    # Load YOLOv11 model
    model = YOLO(f'yolo11{model_size}.pt')
    
    results = model.train(
        data=data_yaml_path,
        epochs=150,          # ← Changed from 100
        imgsz=img_size,
        batch=8,             # ← Changed from 16
        device=device,
        
        # Early stopping
        patience=30,         # ← Changed from 20
        
        # Save settings (keep same)
        save=True,
        save_period=10,
        project='connector_inspection',
        name=f'yolo11{model_size}_single_stage',
        exist_ok=True,
        
        # Optimizer settings
        optimizer='AdamW',
        lr0=0.0005,          # ← Changed from 0.001
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Augmentations (KEEP ALL YOUR CURRENT ONES - they're perfect!)
        degrees=30,
        translate=0.2,
        scale=0.5,
        perspective=0.001,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        
        # Advanced augmentations
        mosaic=1.0,
        mixup=0.15,          # ← Changed from 0.0 (IMPORTANT!)
        copy_paste=0.0,
        
        # Regularization (keep same)
        label_smoothing=0.1,
        
        # Loss weights (keep same)
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Validation
        val=True,
        plots=True,
        
        # For small dataset
        close_mosaic=10,
    )
    
    print(f"\n✅ Training complete!")
    
    # Get best model path
    best_model_path = Path('connector_inspection') / f'yolo11{model_size}_single_stage' / 'weights' / 'best.pt'
    print(f"📁 Best model saved to: {best_model_path}")
    
    # Load best model for validation
    best_model = YOLO(best_model_path)
    
    # Detailed validation
    print("\n📊 Validating on test set...")
    metrics = best_model.val()
    
    print(f"\n📈 Overall Performance:")
    print(f"   mAP@0.5: {metrics.box.map50:.3f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"   Precision: {metrics.box.mp:.3f}")
    print(f"   Recall: {metrics.box.mr:.3f}")
    
    print(f"\n📊 Per-Class Performance:")
    print(f"   Connector OK (class 0):")
    print(f"      - Precision: {metrics.box.p[0]:.3f}")
    print(f"      - Recall: {metrics.box.r[0]:.3f}")
    print(f"      - AP@0.5: {metrics.box.ap50[0]:.3f}")
    
    print(f"   Connector NOT_OK (class 1):")
    print(f"      - Precision: {metrics.box.p[1]:.3f}")
    print(f"      - Recall: {metrics.box.r[1]:.3f}")
    print(f"      - AP@0.5: {metrics.box.ap50[1]:.3f}")
    
    # Export for production
    print("\n📦 Exporting model for deployment...")
    
    # ONNX format (universal)
    best_model.export(format='onnx', simplify=True)
    print("   ✓ ONNX format exported")
    
    # TensorRT format (for NVIDIA GPUs - fastest)
    try:
        best_model.export(format='engine', half=True)  # FP16 precision
        print("   ✓ TensorRT format exported")
    except Exception as e:
        print(f"   ⚠️  TensorRT export failed (requires TensorRT installed): {e}")
    
    print("\n🎯 Next steps:")
    print("   1. Review training plots in: connector_inspection/yolo11{model_size}_single_stage/")
    print("   2. Test inference: python test_inference.py")
    print("   3. Deploy to production")
    
    return best_model

def quick_test(model_path, test_image_path):
    """Quick test inference on a single image."""
    
    model = YOLO(model_path)
    
    print(f"\n🧪 Testing inference on: {test_image_path}")
    
    results = model(test_image_path, conf=0.25)
    
    for result in results:
        boxes = result.boxes
        
        if len(boxes) == 0:
            print("   ⚠️  No connector detected!")
        else:
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                class_name = "connector_ok" if class_id == 0 else "connector_not_ok"
                confidence = float(box.conf[0])
                
                print(f"   Detection {i+1}:")
                print(f"      Class: {class_name}")
                print(f"      Confidence: {confidence:.2%}")
        
        # Save annotated image
        output_path = 'test_inference_result.jpg'
        result.save(output_path)
        print(f"   💾 Saved annotated image to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv11 for single-stage connector inspection')
    
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Training epochs (default: 150)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size (default: 640)')
    parser.add_argument('--model', type=str, default='m',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (default: m - medium)')
    parser.add_argument('--device', default=0,
                        help='GPU device (0 for first GPU, cpu for CPU)')
    parser.add_argument('--test-image', type=str, default=None,
                        help='Test image path for quick inference test')
    
    args = parser.parse_args()
    
    # Train model
    best_model = train_yolo11_single_stage(
        data_yaml_path=args.data,
        epochs=args.epochs,
        img_size=args.img_size,
        model_size=args.model,
        device=args.device
    )
    
    # Quick test if image provided
    if args.test_image:
        model_path = f'connector_inspection/yolo11{args.model}_single_stage/weights/best.pt'
        quick_test(model_path, args.test_image)