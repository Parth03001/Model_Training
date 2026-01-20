from pathlib import Path
import shutil

# ============================================
# COMBINE ALL AUTO-ANNOTATED DATASETS
# ============================================

print("🚀 Combining all datasets into one...\n")

# Paths to your 4 auto-annotated outputs
dataset_paths = [
    'D:\\Training_Scripts\\Training_Scripts\\auto_annotated_black_rim_black_cap',
    'D:\\Training_Scripts\\Training_Scripts\\auto_annotated_black_rim_grey_cap',
    'D:\\Training_Scripts\\Training_Scripts\\auto_annotated_grey_rim_grey_cap',
    'D:\\Training_Scripts\\Training_Scripts\\auto_annotated_grey_rim_black_cap'
]

# Create combined dataset structure
combined_path = Path('./combined_final_dataset')
(combined_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
(combined_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)

total_images = 0
dataset_stats = {}

for dataset_dir in dataset_paths:
    dataset = Path(dataset_dir)
    
    if not dataset.exists():
        print(f"⚠️ Skipping {dataset_dir} (not found)")
        continue
    
    print(f"Processing: {dataset_dir}")
    
    images_dir = dataset / 'images'
    labels_dir = dataset / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"  ⚠️ Missing images or labels folder, skipping")
        continue
    
    count = 0
    for img_file in images_dir.glob('*'):
        if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            # Copy image
            dest_img = combined_path / 'images' / 'train' / img_file.name
            
            # Handle duplicates by adding suffix
            if dest_img.exists():
                base_name = img_file.stem
                suffix = img_file.suffix
                counter = 1
                while dest_img.exists():
                    new_name = f"{base_name}_{counter}{suffix}"
                    dest_img = combined_path / 'images' / 'train' / new_name
                    counter += 1
                
                # Copy with new name
                shutil.copy(img_file, dest_img)
                
                # Copy label with matching new name
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    dest_label = combined_path / 'labels' / 'train' / f"{dest_img.stem}.txt"
                    shutil.copy(label_file, dest_label)
            else:
                # Copy normally
                shutil.copy(img_file, dest_img)
                
                # Copy corresponding label
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    dest_label = combined_path / 'labels' / 'train' / f"{img_file.stem}.txt"
                    shutil.copy(label_file, dest_label)
            
            count += 1
    
    dataset_stats[dataset_dir] = count
    print(f"  ✅ Added {count} images")
    total_images += count

print(f"\n{'='*60}")
print(f"✅ Dataset combination complete!")
print(f"{'='*60}")
print(f"\nDataset breakdown:")
for dataset, count in dataset_stats.items():
    print(f"  {Path(dataset).name}: {count} images")
print(f"\n📊 Total combined images: {total_images}")

# Create data.yaml for training
with open(combined_path / 'data.yaml', 'w') as f:
    f.write(f"""path: {combined_path.absolute()}
train: images/train
val: images/train

nc: 4
names:
  - rim_black
  - cap_black
  - rim_grey
  - cap_grey
""")

print(f"\n✅ data.yaml created")
print(f"📁 Combined dataset location: {combined_path}/")
print(f"\nDataset structure:")
print(f"  combined_final_dataset/")
print(f"  ├── images/")
print(f"  │   └── train/       ({total_images} images)")
print(f"  ├── labels/")
print(f"  │   └── train/       ({total_images} labels)")
print(f"  └── data.yaml")

# Create class distribution report
class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
labels_path = combined_path / 'labels' / 'train'

for label_file in labels_path.glob('*.txt'):
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                cls = int(parts[0])
                class_counts[cls] = class_counts.get(cls, 0) + 1

print(f"\n📊 Class distribution:")
print(f"  rim_black (0): {class_counts[0]} instances")
print(f"  cap_black (1): {class_counts[1]} instances")
print(f"  rim_grey  (2): {class_counts[2]} instances")
print(f"  cap_grey  (3): {class_counts[3]} instances")

print(f"\n✅ Ready for training!")
print(f"\n👉 Next step: Train the final model with this combined dataset")