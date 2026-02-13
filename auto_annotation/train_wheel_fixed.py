from ultralytics import YOLO
from pathlib import Path

# ============================================ 
# FIXED TRAINING FOR COLOR SENSITIVITY
# ============================================ 

if __name__ == '__main__':
    # Dataset path
    data_yaml = 'D:\\Training_Scripts\\Scripts\\combined_final_dataset\\data.yaml'
    
    # Model config
    model_size = 'm'
    project_dir = './wheel_training_fixed'
    run_name = 'yolo26m_color_sensitive'
    
    print(f"\n{'='*60}")
    print(f"STARTING COLOR-SENSITIVE TRAINING")
    print(f"{'='*60}")
    print("KEY CHANGE: Disabling HSV/Color augmentations to preserve Black vs Grey distinction.")

    # Load model
    model = YOLO(f'yolo26{model_size}.pt') 

    results = model.train(
        data=data_yaml,
        epochs=100,      # 100 epochs is usually enough for fine-tuning
        patience=20,
        imgsz=640,
        batch=8,
        device=0,

        # ============================================ 
        # CRITICAL FIXES FOR COLOR CLASSIFICATION
        # ============================================ 
        
        # 1. DISABLE COLOR AUGMENTATION
        # We want the model to learn the EXACT color of the rim.
        # Changing saturation/brightness makes Black look like Grey.
        hsv_h=0.0,       # Disable hue change
        hsv_s=0.0,       # Disable saturation change (CRITICAL)
        hsv_v=0.0,       # Disable brightness change (CRITICAL)
        
        # 2. Geometric Augmentations (Safe to keep)
        degrees=10,      # Reduced rotation slightly
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        
        # 3. Mixing/Regularization
        mosaic=1.0,      # Mosaic is fine (spatial only)
        mixup=0.0,       # DISABLE MIXUP: Blending images blends colors!
        copy_paste=0.0,
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.0005,
        
        # Output
        project=project_dir,
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )
    
    print(f"Training Complete. Check {project_dir}/{run_name}")
