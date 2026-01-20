"""
YOLO Dataset Diagnostic Tool - Python Version
Checks for common issues with image paths and dataset organization
"""
import os
from pathlib import Path

def check_source_directory(source_dir):
    """Check the source directory with annotated images"""
    print("\n" + "="*50)
    print("Check 1: Source Directory")
    print("="*50)
    
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"❌ Source directory NOT FOUND: {source_dir}")
        return False
    
    print(f"✓ Source directory exists: {source_dir}")
    
    # Count images
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = [f for f in source_path.iterdir() 
              if f.is_file() and f.suffix in image_extensions]
    
    # Count labels
    labels = [f for f in source_path.iterdir() 
              if f.is_file() and f.suffix == '.txt']
    
    print(f"  Images found: {len(images)}")
    print(f"  Labels found: {len(labels)}")
    
    # Check for spaces in filenames
    files_with_spaces = [f for f in source_path.iterdir() 
                        if f.is_file() and ' ' in f.name]
    
    if files_with_spaces:
        print(f"  ⚠ WARNING: {len(files_with_spaces)} files have SPACES in names!")
        print("  Example files with spaces:")
        for f in files_with_spaces[:3]:
            print(f"    - {f.name}")
        return False
    else:
        print("  ✓ No spaces in filenames")
    
    if len(images) != len(labels):
        print(f"  ⚠ WARNING: Image count ({len(images)}) != Label count ({len(labels)})")
        return False
    
    return True

def check_organized_dataset(yolo_dir):
    """Check the organized YOLO dataset structure"""
    print("\n" + "="*50)
    print("Check 2: Organized Dataset")
    print("="*50)
    
    yolo_path = Path(yolo_dir)
    
    if not yolo_path.exists():
        print(f"❌ YOLO dataset NOT organized yet: {yolo_dir}")
        print("  Run: python3 organise_annotation.py")
        return False
    
    print(f"✓ YOLO dataset directory exists: {yolo_dir}")
    
    # Check structure
    issues = []
    
    # Check train
    train_imgs = yolo_path / 'images' / 'train'
    train_lbls = yolo_path / 'labels' / 'train'
    
    if train_imgs.exists():
        img_count = len(list(train_imgs.glob('*.*')))
        print(f"  ✓ Train images: {img_count}")
    else:
        print("  ❌ Train images directory missing!")
        issues.append("train_images_missing")
    
    if train_lbls.exists():
        lbl_count = len(list(train_lbls.glob('*.txt')))
        print(f"  ✓ Train labels: {lbl_count}")
    else:
        print("  ❌ Train labels directory missing!")
        issues.append("train_labels_missing")
    
    # Check val
    val_imgs = yolo_path / 'images' / 'val'
    val_lbls = yolo_path / 'labels' / 'val'
    
    if val_imgs.exists():
        img_count = len(list(val_imgs.glob('*.*')))
        print(f"  ✓ Val images: {img_count}")
    else:
        print("  ❌ Val images directory missing!")
        issues.append("val_images_missing")
    
    if val_lbls.exists():
        lbl_count = len(list(val_lbls.glob('*.txt')))
        print(f"  ✓ Val labels: {lbl_count}")
    else:
        print("  ❌ Val labels directory missing!")
        issues.append("val_labels_missing")
    
    return len(issues) == 0

def check_data_yaml(yolo_dir):
    """Check data.yaml configuration"""
    print("\n" + "="*50)
    print("Check 3: data.yaml")
    print("="*50)
    
    yaml_path = Path(yolo_dir) / 'data.yaml'
    
    if not yaml_path.exists():
        print(f"❌ data.yaml NOT FOUND at {yaml_path}")
        return False
    
    print("✓ data.yaml exists")
    print("\nContents:")
    
    with open(yaml_path, 'r') as f:
        content = f.read()
        print(content)
    
    # Check for correct format
    if 'yolo_dataset' in content or yolo_dir in content:
        print("  ✓ Path looks correct")
    else:
        print("  ⚠ Path might be incorrect")
    
    if 'train: images/train' in content:
        print("  ✓ Train path is relative (correct)")
    else:
        print("  ⚠ Train path should be relative: 'images/train'")
    
    return True

def check_image_label_pairing(yolo_dir):
    """Check if images and labels are properly paired"""
    print("\n" + "="*50)
    print("Check 4: Image-Label Pairing")
    print("="*50)
    
    train_imgs = Path(yolo_dir) / 'images' / 'train'
    train_lbls = Path(yolo_dir) / 'labels' / 'train'
    
    if not train_imgs.exists():
        print("❌ Train images directory not found")
        return False
    
    # Get first image
    images = list(train_imgs.glob('*.*'))
    if not images:
        print("❌ No images found in train directory")
        return False
    
    sample_img = images[0]
    print(f"Sample image: {sample_img.name}")
    
    # Check corresponding label
    sample_label = train_lbls / f"{sample_img.stem}.txt"
    
    if sample_label.exists():
        print(f"✓ Corresponding label exists: {sample_label.name}")
        print("  Label content:")
        with open(sample_label, 'r') as f:
            print(f"    {f.read().strip()}")
        return True
    else:
        print(f"❌ Corresponding label NOT FOUND: {sample_label.name}")
        return False

def main():
    """Main diagnostic function"""
    print("="*50)
    print("YOLO Dataset Diagnostic Tool")
    print("="*50)
    
    # Configuration
    SOURCE_DIR = "C:\\Users\\50014665\\Image_Classification\\all_images"
    YOLO_DIR = "C:\\Users\\50014665\Image_Classification\\yolo_dataset"

    # Run checks
    issues = []
    
    if not check_source_directory(SOURCE_DIR):
        issues.append("source_directory")
    
    if not check_organized_dataset(YOLO_DIR):
        issues.append("organized_dataset")
    
    if not check_data_yaml(YOLO_DIR):
        issues.append("data_yaml")
    
    if not check_image_label_pairing(YOLO_DIR):
        issues.append("image_label_pairing")
    
    # Summary
    print("\n" + "="*50)
    print("DIAGNOSTIC SUMMARY")
    print("="*50)
    
    if not issues:
        print("✓ No major issues detected!")
        print("\nIf you're still getting errors, try:")
        print("  python3 fix_paths.py")
    else:
        print(f"❌ Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRun fix script:")
        print("  python3 fix_paths.py")
    
    print("="*50)

if __name__ == "__main__":
    main()