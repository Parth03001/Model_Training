import shutil
from pathlib import Path
import random

dataset = Path('D:\\Training_Scripts\\Scripts\\combined_final_dataset')

# Create val directories
(dataset / 'images' / 'val').mkdir(exist_ok=True)
(dataset / 'labels' / 'val').mkdir(exist_ok=True)

# Get all images
all_images = list((dataset / 'images' / 'train').glob('*.jpg'))
random.seed(42)
random.shuffle(all_images)

# Move 20% to validation (1278 images)
val_count = int(len(all_images) * 0.2)

for img in all_images[:val_count]:
    # Move image
    shutil.move(img, dataset / 'images' / 'val' / img.name)
    
    # Move label
    label_src = dataset / 'labels' / 'train' / f"{img.stem}.txt"
    label_dst = dataset / 'labels' / 'val' / f"{img.stem}.txt"
    if label_src.exists():
        shutil.move(label_src, label_dst)

print(f"✅ Train: {len(all_images) - val_count}")
print(f"✅ Val: {val_count}")