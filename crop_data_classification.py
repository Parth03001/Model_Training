"""
Stage 1.5: Use YOLO to Crop Connectors from Original Images
"""
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil

def crop_connectors_from_dataset(
    yolo_model_path,
    original_data_dir,
    output_dir,
    conf_threshold=0.25
):
    """
    Use trained YOLO to detect and crop connectors from OK/NOT_OK dataset
    
    Args:
        yolo_model_path: Path to trained YOLO weights (best.pt)
        original_data_dir: Original data dir with ok/ and not_ok/ folders
        output_dir: Output directory for cropped images
        conf_threshold: Detection confidence threshold
    """
    
    # Load YOLO model
    model = YOLO(yolo_model_path)
    print(f"✓ Loaded YOLO detector from {yolo_model_path}")
    
    original_data_dir = Path(original_data_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    (output_dir / 'ok').mkdir(parents=True, exist_ok=True)
    (output_dir / 'not_ok').mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_name in ['ok', 'not_ok']:
        input_folder = original_data_dir / class_name
        output_folder = output_dir / class_name
        
        if not input_folder.exists():
            print(f"⚠ Skipping {class_name} - folder not found")
            continue
        
        image_files = list(input_folder.glob('*.jpg')) + list(input_folder.glob('*.jpeg')) + list(input_folder.glob('*.png'))
        
        print(f"\n📦 Processing {class_name}: {len(image_files)} images")
        
        cropped_count = 0
        no_detection_count = 0
        
        for img_path in tqdm(image_files, desc=f"Cropping {class_name}"):
            # Run detection
            results = model(img_path, conf=conf_threshold, verbose=False)
            
            # Get image
            img = Image.open(img_path).convert('RGB')
            
            # Check if connector detected
            if len(results[0].boxes) == 0:
                # No detection - save original image with warning
                print(f"⚠ No detection in {img_path.name} - saving original")
                img.save(output_folder / img_path.name)
                no_detection_count += 1
                continue
            
            # Get first (highest confidence) detection
            box = results[0].boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Add padding (10% on each side)
            width = x2 - x1
            height = y2 - y1
            padding_w = width * 0.1
            padding_h = height * 0.1
            
            x1 = max(0, x1 - padding_w)
            y1 = max(0, y1 - padding_h)
            x2 = min(img.width, x2 + padding_w)
            y2 = min(img.height, y2 + padding_h)
            
            # Crop connector region
            cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))
            
            # Save cropped image
            cropped.save(output_folder / img_path.name)
            cropped_count += 1
        
        print(f"✓ {class_name}: {cropped_count} cropped, {no_detection_count} no detection")
    
    print(f"\n✓ All connectors cropped and saved to: {output_dir}")
    print(f"\n📊 Summary:")
    print(f"  Cropped images saved to: {output_dir}")
    print(f"  Next step: Train classifier on cropped images")

def verify_cropped_dataset(cropped_data_dir):
    """Verify cropped dataset structure"""
    cropped_data_dir = Path(cropped_data_dir)
    
    ok_count = len(list((cropped_data_dir / 'ok').glob('*')))
    not_ok_count = len(list((cropped_data_dir / 'not_ok').glob('*')))
    
    print(f"\n📊 Cropped Dataset Verification:")
    print(f"  OK images: {ok_count}")
    print(f"  NOT_OK images: {not_ok_count}")
    print(f"  Total: {ok_count + not_ok_count}")
    
    return ok_count, not_ok_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=str, required=True, 
                        help='Path to trained YOLO model (best.pt)')
    parser.add_argument('--original-data', type=str, required=True,
                        help='Path to original data with ok/not_ok folders')
    parser.add_argument('--output-dir', type=str, default='cropped_connectors',
                        help='Output directory for cropped images')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    crop_connectors_from_dataset(
        args.yolo_model,
        args.original_data,
        args.output_dir,
        args.conf
    )
    
    verify_cropped_dataset(args.output_dir)