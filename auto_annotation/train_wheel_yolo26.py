from ultralytics import YOLO
from pathlib import Path

# ============================================
# YOLO26 TRAINING CONFIGURATION
# ============================================
# YOLO26 Key Features (Released January 2026):
# - End-to-End NMS-Free inference (faster deployment)
# - DFL Removal (simpler exports, better edge compatibility)
# - MuSGD Optimizer (hybrid SGD + Muon, inspired by LLM training)
# - Up to 43% faster CPU inference vs YOLO11
# - ProgLoss + STAL (better small object detection)
# ============================================

if __name__ == '__main__':
    # Dataset path
    data_yaml = 'D:\\Training_Scripts\\Scripts\\combined_final_dataset\\data.yaml'

    # ============================================
    # MODEL SIZE SELECTION FOR ~6000 IMAGES, 4 CLASSES
    # ============================================
    # YOLO26 Model Variants:
    # 'n' = Nano  (~2.4M params, 40.9 mAP) - edge devices, fastest
    # 's' = Small (~9.5M params, 48.6 mAP) - good balance for small datasets
    # 'm' = Medium (~20M params, ~51.5 mAP) - RECOMMENDED for your dataset
    # 'l' = Large (~25M params, ~53.4 mAP) - if you need higher accuracy
    # 'x' = XLarge (~68M params) - overkill for 4 classes
    #
    # RECOMMENDATION FOR YOUR USE CASE:
    # - Dataset: ~6000 images, 4 classes (rim_black, cap_black, rim_grey, cap_grey)
    # - Model Size: Medium ('m') is optimal
    #   * Small ('s') can work if inference speed is priority
    #   * Large ('l') if you need maximum accuracy and have GPU resources
    # - Rule of thumb: Start with 's' or 'm', scale up if mAP plateaus
    # ============================================

    model_size = 'm'  # RECOMMENDED for ~6000 images with 4 classes

    # Training output directory
    project_dir = './wheel_training_yolo26'
    run_name = 'yolo26m_4class_optimized'
    checkpoint_path = Path(project_dir) / run_name / 'weights' / 'last.pt'

    print(f"\n{'='*60}")
    print(f"TRAINING YOLO26-{model_size.upper()} ON 4-CLASS WHEEL DATASET")
    print(f"{'='*60}")
    print(f"\nDataset: ~6000 images")
    print(f"Classes: rim_black, cap_black, rim_grey, cap_grey")
    print(f"\nYOLO26 Advantages over YOLO11:")
    print(f"  - End-to-End NMS-Free inference")
    print(f"  - Up to 43% faster CPU inference")
    print(f"  - Better small object detection (STAL)")
    print(f"  - Simpler model export for edge devices")

    # ============================================
    # AUTO-RESUME FUNCTIONALITY
    # ============================================
    resume_training = False
    if checkpoint_path.exists():
        print(f"\n[RESUME MODE] Found checkpoint at {checkpoint_path}")
        print(f"   Resuming training from previous session...")
        model = YOLO(checkpoint_path)
        resume_training = True
    else:
        print(f"\n[FRESH START] No checkpoint found")
        print(f"   Starting new training from yolo26{model_size}.pt pretrained weights")
        model = YOLO(f'yolo26{model_size}.pt')  # Load YOLO26 pretrained weights

    print()

    # ============================================
    # TRAINING CONFIGURATION
    # ============================================
    # NOTE: Keeping same optimized parameters from YOLO11 to avoid overfitting
    # Key anti-overfitting measures:
    # - Reduced learning rate (0.0005 instead of 0.001)
    # - Smaller batch size (8 instead of 32)
    # - Disabled copy_paste augmentation
    # - Reduced mixup (0.15 instead of 0.3)
    # - Label smoothing enabled
    # ============================================

    results = model.train(
        # Resume from checkpoint if exists
        resume=resume_training,

        # Dataset
        data=data_yaml,

        # Training duration
        epochs=150,
        patience=30,  # Early stopping if no improvement

        # Image settings
        imgsz=640,
        batch=8,  # REDUCED - smaller batches improve generalization

        # Device
        device=0,  # Use GPU 0, change to 'cpu' if no GPU

        # ============================================
        # OPTIMIZER SETTINGS
        # ============================================
        # YOLO26 introduces MuSGD optimizer, but AdamW still works well
        # for transfer learning and fine-tuning tasks
        optimizer='AdamW',
        lr0=0.0005,     # Initial learning rate (REDUCED to prevent overfitting)
        lrf=0.01,       # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,

        # ============================================
        # AGGRESSIVE AUGMENTATION (Optimized for wheels)
        # ============================================

        # Geometric augmentations
        degrees=30,        # Rotate +/-30 degrees (wheels can be at angles)
        translate=0.2,     # Translate +/-20%
        scale=0.5,         # Scale 50-150%
        shear=5,           # Shear +/-5 degrees
        perspective=0.001, # Perspective distortion
        flipud=0.0,        # No vertical flip (wheels don't flip vertically)
        fliplr=0.5,        # 50% horizontal flip

        # Color augmentations (IMPORTANT for grey vs black distinction)
        hsv_h=0.02,        # Hue variation +/-2%
        hsv_s=0.7,         # Saturation variation +/-70%
        hsv_v=0.4,         # Brightness variation +/-40%

        # Advanced augmentations
        mosaic=1.0,        # 100% mosaic (combine 4 images)
        mixup=0.15,        # REDUCED from 0.3 - less aggressive blending
        copy_paste=0.0,    # DISABLED - was causing overfitting

        # Regularization
        label_smoothing=0.1,

        # ============================================
        # LOSS WEIGHTS
        # ============================================
        # NOTE: YOLO26 removed DFL (Distribution Focal Loss)
        # so we only configure box and cls loss weights
        box=7.5,           # Box loss weight
        cls=0.5,           # Class loss weight
        # dfl parameter removed in YOLO26 - DFL is no longer used

        # Mosaic settings
        close_mosaic=10,   # Disable mosaic last 10 epochs

        # Validation
        val=True,
        plots=True,

        # Save settings
        save=True,
        save_period=10,    # Save every 10 epochs (creates checkpoints)
        project=project_dir,
        name=run_name,
        exist_ok=True,

        # Performance
        workers=8,
        verbose=True
    )

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nModel saved to: {checkpoint_path.parent / 'best.pt'}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"\nTIP: If training was interrupted, simply run this script again")
    print(f"   It will automatically resume from the last checkpoint!")
    print(f"\n{'='*60}")
    print(f"YOLO26 vs YOLO11 Expected Improvements:")
    print(f"{'='*60}")
    print(f"  - Faster inference (up to 43% on CPU)")
    print(f"  - Better edge deployment (no NMS required)")
    print(f"  - Simpler ONNX/TensorRT export")
    print(f"  - Better small object detection with STAL")
