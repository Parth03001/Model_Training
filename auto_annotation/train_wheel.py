from ultralytics import YOLO
from pathlib import Path

# ============================================
# TRAINING CONFIGURATION
# ============================================
if __name__ == '__main__':
    # Dataset path
    data_yaml = 'D:\\Training_Scripts\\Scripts\\combined_final_dataset\\data.yaml'

    # Model selection - CHOOSE ONE:
    # 'n' = Nano (fastest, ~3M params) - for edge devices
    # 's' = Small (~9M params) - good balance
    # 'm' = Medium (~20M params) - RECOMMENDED for your dataset
    # 'l' = Large (~25M params) - if you need maximum accuracy
    # 'x' = XLarge (~68M params) - overkill for this task

    model_size = 'm'  # RECOMMENDED for 6388 images

    print(f"\n{'='*60}")
    print(f"🚀 TRAINING YOLO11-{model_size.upper()} ON 4-CLASS DATASET")
    print(f"{'='*60}")
    print(f"\nDataset: 6388 images")
    print(f"Classes: rim_black, cap_black, rim_grey, cap_grey")
    print(f"Model: yolo11{model_size}.pt\n")

    # ============================================
    # OPTION A: AGGRESSIVE AUGMENTATION (RECOMMENDED)
    # ============================================

    model = YOLO(f'yolo11{model_size}.pt')

    results = model.train(
        # Dataset
        data=data_yaml,
        
        # Training duration
        epochs=150,
        patience=30,  # Early stopping if no improvement
        
        # Image settings
        imgsz=640,
        batch=8,  # REDUCED from 32 - smaller batches improve generalization
        
        # Device
        device=0,  # Use GPU 0, change to 'cpu' if no GPU
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.0005,     # Initial learning rate (REDUCED from 0.001 - prevents overfitting)
        lrf=0.01,       # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        
        # ============================================
        # AGGRESSIVE AUGMENTATION (Perfect for your dataset)
        # ============================================
        
        # Geometric augmentations
        degrees=30,        # Rotate ±30 degrees (wheels can be at angles)
        translate=0.2,     # Translate ±20%
        scale=0.5,         # Scale 50-150%
        shear=5,           # Shear ±5 degrees
        perspective=0.001, # Perspective distortion
        flipud=0.0,        # No vertical flip (wheels don't flip)
        fliplr=0.5,        # 50% horizontal flip
        
        # Color augmentations (IMPORTANT for grey vs black)
        hsv_h=0.02,        # Hue variation ±2%
        hsv_s=0.7,         # Saturation variation ±70%
        hsv_v=0.4,         # Brightness variation ±40%
        
        # Advanced augmentations
        mosaic=1.0,        # 100% mosaic (combine 4 images)
        mixup=0.15,        # REDUCED from 0.3 - less aggressive blending
        copy_paste=0.0,    # DISABLED - was causing overfitting
        
        # Regularization
        label_smoothing=0.1,
        # erasing=0.4,     # REMOVED - was too aggressive and caused overfitting
        
        # Loss weights
        box=7.5,           # Box loss weight
        cls=0.5,           # Class loss weight
        dfl=1.5,           # Distribution focal loss
        
        # Mosaic settings
        close_mosaic=10,   # Disable mosaic last 10 epochs
        
        # Validation
        val=True,
        plots=True,
        
        # Save settings
        save=True,
        save_period=10,    # Save every 10 epochs
        project='./wheel_training',
        name='yolo11m_4class_aggressive_aug',
        exist_ok=True,
        
        # Performance
        workers=8,
        verbose=True
    )

    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*60}")

