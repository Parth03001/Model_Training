"""
Organize LabelImg annotations into YOLO training structure
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def organize_yolo_dataset(
    source_dir,
    output_dir='yolo_dataset',
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1
):
    """
    Organize annotated images and labels into YOLO format
    
    Args:
        source_dir: Directory with images and .txt labels
        output_dir: Output directory for organized dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    """
    
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(list(source_dir.glob(f'*{ext}')))
    
    print(f"Found {len(all_images)} images in {source_dir}")
    
    # Filter images that have corresponding label files
    images_with_labels = []
    for img_path in all_images:
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            images_with_labels.append(img_path)
        else:
            print(f"⚠ Warning: No label for {img_path.name}")
    
    print(f"Images with labels: {len(images_with_labels)}")
    
    if len(images_with_labels) == 0:
        print("❌ Error: No images with labels found!")
        return
    
    # Split dataset
    train_images, temp_images = train_test_split(
        images_with_labels, 
        test_size=(1 - train_ratio),
        random_state=42
    )
    
    val_images, test_images = train_test_split(
        temp_images,
        test_size=test_ratio/(val_ratio + test_ratio),
        random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_images)} images ({len(train_images)/len(images_with_labels)*100:.1f}%)")
    print(f"  Val:   {len(val_images)} images ({len(val_images)/len(images_with_labels)*100:.1f}%)")
    print(f"  Test:  {len(test_images)} images ({len(test_images)/len(images_with_labels)*100:.1f}%)")
    
    # Create directory structure
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    for split_name, image_list in splits.items():
        # Create directories
        img_dir = output_dir / 'images' / split_name
        label_dir = output_dir / 'labels' / split_name
        
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for img_path in image_list:
            label_path = img_path.with_suffix('.txt')
            
            # Copy image
            shutil.copy(img_path, img_dir / img_path.name)
            
            # Copy label
            shutil.copy(label_path, label_dir / label_path.name)
        
        print(f"✓ Created {split_name} set: {len(image_list)} images")
    
    # Create data.yaml
    yaml_content = f"""# Connector Detection Dataset
path: {output_dir.absolute()}  # dataset root dir
train: images/train
val: images/val
test: images/test

# Classes
nc: 1  # number of classes
names: ['connector']  # class names
"""
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Created data.yaml at {yaml_path}")
    print(f"\n✅ Dataset organized successfully!")
    print(f"\nDirectory structure:")
    print(f"{output_dir}/")
    print(f"├── images/")
    print(f"│   ├── train/   ({len(train_images)} images)")
    print(f"│   ├── val/     ({len(val_images)} images)")
    print(f"│   └── test/    ({len(test_images)} images)")
    print(f"├── labels/")
    print(f"│   ├── train/   ({len(train_images)} labels)")
    print(f"│   ├── val/     ({len(val_images)} labels)")
    print(f"│   └── test/    ({len(test_images)} labels)")
    print(f"└── data.yaml")
    print(f"\n🚀 Next step: Train YOLO detector")
    print(f"   python stage1_train_detector.py --data {yaml_path}")
    
    return output_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Organize LabelImg annotations for YOLO training")
    parser.add_argument('--source', type=str, default='all_images',
                        help='Source directory with images and .txt labels')
    parser.add_argument('--output', type=str, default='yolo_dataset',
                        help='Output directory for organized dataset')
    parser.add_argument('--train', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.2,
                        help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train + args.val + args.test - 1.0) > 0.001:
        print("❌ Error: train + val + test must equal 1.0")
        exit(1)
    
    organize_yolo_dataset(
        args.source,
        args.output,
        args.train,
        args.val,
        args.test
    )