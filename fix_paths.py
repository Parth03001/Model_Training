"""
YOLO Dataset Fix Script - Python Version
Automatically fixes common issues with image paths and dataset organization
"""
import os
import shutil
from pathlib import Path
import subprocess

def remove_spaces_from_filenames(source_dir):
    """Remove spaces from all filenames in source directory"""
    print("\n" + "="*50)
    print("Step 1: Removing Spaces from Filenames")
    print("="*50)
    
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return False
    
    renamed_count = 0
    
    # Get all files with spaces
    for file_path in source_path.iterdir():
        if file_path.is_file() and ' ' in file_path.name:
            # Create new name without spaces
            new_name = file_path.name.replace(' ', '_')
            new_path = file_path.parent / new_name
            
            # Rename
            try:
                file_path.rename(new_path)
                print(f"  Renamed: {file_path.name} -> {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"  ❌ Error renaming {file_path.name}: {e}")
    
    if renamed_count > 0:
        print(f"\n✓ Renamed {renamed_count} files")
    else:
        print("\n✓ No files needed renaming")
    
    return True

def count_files(source_dir):
    """Count images and labels in source directory"""
    print("\n" + "="*50)
    print("Step 2: Counting Files")
    print("="*50)
    
    source_path = Path(source_dir)
    
    # Count images
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = [f for f in source_path.iterdir() 
              if f.is_file() and f.suffix in image_extensions]
    
    # Count labels
    labels = [f for f in source_path.iterdir() 
              if f.is_file() and f.suffix == '.txt']
    
    print(f"Images found: {len(images)}")
    print(f"Labels found: {len(labels)}")
    
    if len(images) != len(labels):
        print(f"\n⚠ WARNING: Image count ({len(images)}) != Label count ({len(labels)})")
        print("Some images may not have labels!")
        
        # Find images without labels
        missing_labels = []
        for img in images:
            label_path = source_path / f"{img.stem}.txt"
            if not label_path.exists():
                missing_labels.append(img.name)
        
        if missing_labels:
            print(f"\nImages without labels ({len(missing_labels)}):")
            for img_name in missing_labels[:5]:
                print(f"  - {img_name}")
            if len(missing_labels) > 5:
                print(f"  ... and {len(missing_labels) - 5} more")
    else:
        print("\n✓ All images have corresponding labels")
    
    return len(images), len(labels)

def reorganize_dataset(source_dir, output_dir):
    """Reorganize dataset using organize_annotations.py"""
    print("\n" + "="*50)
    print("Step 3: Reorganizing Dataset")
    print("="*50)
    
    output_path = Path(output_dir)
    
    # Remove old organized data if exists
    if output_path.exists():
        print(f"Removing old dataset: {output_dir}")
        shutil.rmtree(output_path)
        print("✓ Removed old dataset")
    
    # Run organize_annotations.py
    print(f"\nOrganizing dataset...")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    
    try:
        result = subprocess.run([
            'python3', 'organise_annotation.py',
            '--source', source_dir,
            '--output', output_dir
        ], capture_output=True, text=True, check=True)
        
        print(result.stdout)
        print("\n✓ Dataset organized successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error organizing dataset:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("\n❌ organise_annotation.py not found!")
        print("Make sure you're running this from the correct directory")
        return False

def verify_data_yaml(yolo_dir):
    """Verify and fix data.yaml if needed"""
    print("\n" + "="*50)
    print("Step 4: Verifying data.yaml")
    print("="*50)
    
    yaml_path = Path(yolo_dir) / 'data.yaml'
    
    if not yaml_path.exists():
        print(f"❌ data.yaml not found at {yaml_path}")
        return False
    
    print("✓ data.yaml exists")
    print("\nContents:")
    
    with open(yaml_path, 'r') as f:
        content = f.read()
        print(content)
    
    # Check if paths are correct
    correct_format = f"""path: {yolo_dir}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['connector']
"""
    
    if 'train: images/train' not in content:
        print("\n⚠ Fixing data.yaml format...")
        with open(yaml_path, 'w') as f:
            f.write(correct_format)
        print("✓ data.yaml fixed")
    else:
        print("\n✓ data.yaml format is correct")
    
    return True

def verify_structure(yolo_dir):
    """Verify the organized dataset structure"""
    print("\n" + "="*50)
    print("Step 5: Verifying Directory Structure")
    print("="*50)
    
    yolo_path = Path(yolo_dir)
    
    checks = {
        'images/train': 0,
        'images/val': 0,
        'images/test': 0,
        'labels/train': 0,
        'labels/val': 0,
        'labels/test': 0
    }
    
    all_good = True
    
    for subdir, count in checks.items():
        dir_path = yolo_path / subdir
        if dir_path.exists():
            file_count = len(list(dir_path.glob('*.*')))
            print(f"✓ {subdir}: {file_count} files")
        else:
            print(f"❌ {subdir}: NOT FOUND")
            all_good = False
    
    return all_good

def main():
    """Main fix function"""
    print("="*50)
    print("YOLO Dataset Fix Script")
    print("="*50)
    
    # Configuration
    SOURCE_DIR = "C:\\Users\\50014665\\Image_Classification\\all_images"
    OUTPUT_DIR = "C:\\Users\\50014665\\Image_Classification\\yolo_dataset"
    
    # Step 1: Remove spaces
    if not remove_spaces_from_filenames(SOURCE_DIR):
        print("\n❌ Failed to fix filenames")
        return
    
    # Step 2: Count files
    img_count, lbl_count = count_files(SOURCE_DIR)
    
    # Step 3: Reorganize
    if not reorganize_dataset(SOURCE_DIR, OUTPUT_DIR):
        print("\n❌ Failed to reorganize dataset")
        return
    
    # Step 4: Verify data.yaml
    if not verify_data_yaml(OUTPUT_DIR):
        print("\n❌ Failed to verify data.yaml")
        return
    
    # Step 5: Verify structure
    if not verify_structure(OUTPUT_DIR):
        print("\n❌ Directory structure incomplete")
        return
    
    # Summary
    print("\n" + "="*50)
    print("FIX COMPLETE!")
    print("="*50)
    print("\n✓ All issues fixed")
    print("\nNext steps:")
    print("1. Verify with: python3 diagnose_paths.py")
    print("2. Train YOLO: python3 stage1_train_detector.py --data", OUTPUT_DIR + "/data.yaml")
    print("="*50)

if __name__ == "__main__":
    main()