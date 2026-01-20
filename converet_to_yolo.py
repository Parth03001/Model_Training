"""Organize existing YOLO annotations into train/val/test splits"""
import os
import shutil
from pathlib import Path
import random

def organize_yolo_dataset(ok_folder, not_ok_folder, output_folder, train_split=0.80, val_split=0.15):
    """
    Organize OK and NOT_OK folders with YOLO .txt labels
    Updates class IDs: OK=0, NOT_OK=1
    """
    
    ok_path = Path(ok_folder)
    not_ok_path = Path(not_ok_folder)
    output_path = Path(output_folder)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def process_folder(folder_path, class_id):
        """Process one folder and update class IDs"""
        data = []
        
        # Find all images
        for ext in ['.jpg', '.jpeg', '.png']:
            for img_file in folder_path.glob(f'*{ext}'):
                txt_file = img_file.with_suffix('.txt')
                
                if not txt_file.exists():
                    print(f"Warning: Label not found for {img_file.name}")
                    continue
                
                # Read and update class ID
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                
                updated_labels = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # Change first number (class) to our class_id
                        parts[0] = str(class_id)
                        updated_labels.append(' '.join(parts))
                
                data.append((img_file, updated_labels))
        
        return data
    
    # Process both folders
    print("Processing OK folder (class 0)...")
    ok_data = process_folder(ok_path, class_id=0)
    print(f"Found {len(ok_data)} OK images")
    
    print("Processing NOT_OK folder (class 1)...")
    not_ok_data = process_folder(not_ok_path, class_id=1)
    print(f"Found {len(not_ok_data)} NOT_OK images")
    
    # Combine and shuffle
    all_data = ok_data + not_ok_data
    random.shuffle(all_data)
    
    # Split data
    n_total = len(all_data)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train + n_val]
    test_data = all_data[n_train + n_val:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")
    
    # Copy files
    def copy_split(data, split_name):
        for img_file, labels in data:
            # Copy image
            dst_img = output_path / split_name / 'images' / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # Write label
            txt_file = output_path / split_name / 'labels' / img_file.with_suffix('.txt').name
            with open(txt_file, 'w') as f:
                f.write('\n'.join(labels) + '\n')
    
    print("\nCopying files...")
    copy_split(train_data, 'train')
    copy_split(val_data, 'val')
    copy_split(test_data, 'test')
    
    # Create data.yaml
    yaml_content = f"""path: {output_path.absolute()}
train: train/images
val: val/images
test: test/images

nc: 2
names: ['OK', 'NOT_OK']
"""
    
    yaml_file = output_path / 'data.yaml'
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Dataset ready: {output_path.absolute()}")
    print(f"✅ Config: {yaml_file}")

if __name__ == "__main__":
    organize_yolo_dataset(
        ok_folder=r"C:\Users\50014665\Downloads\data\ok condition",
        not_ok_folder=r"C:\Users\50014665\Downloads\data\not ok condition",
        output_folder="round_2_connector_dataset"
    )