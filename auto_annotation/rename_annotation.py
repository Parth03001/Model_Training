import shutil
from pathlib import Path

# Configuration
source_folder = '../Wheel Data/AX7_OK'  # Change this to your folder path
output_folder = './manually_annotated'

# Create output folder
Path(output_folder).mkdir(exist_ok=True)

# Get all annotation files (excluding classes.txt)
source_path = Path(source_folder)
annotation_files = [f for f in source_path.glob('*.txt') if f.name != 'classes.txt']

print(f"Found {len(annotation_files)} annotation files")

copied_count = 0

for txt_file in annotation_files:
    # Handle .xml.txt files
    if txt_file.name.endswith('.xml.txt'):
        base_name = txt_file.name.replace('.xml.txt', '')
        new_txt_name = base_name + '.txt'
    else:
        base_name = txt_file.stem
        new_txt_name = txt_file.name
    
    # Find corresponding image
    image_file = None
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        potential_image = source_path / f"{base_name}{ext}"
        if potential_image.exists():
            image_file = potential_image
            break
    
    if image_file:
        # Copy image
        shutil.copy(image_file, Path(output_folder) / image_file.name)
        
        # Copy and rename annotation
        shutil.copy(txt_file, Path(output_folder) / new_txt_name)
        
        copied_count += 1
        print(f"Copied: {image_file.name} + {new_txt_name}")

# Copy classes.txt if exists
classes_file = source_path / 'classes.txt'
if classes_file.exists():
    shutil.copy(classes_file, Path(output_folder) / 'classes.txt')
    print("Copied: classes.txt")

print(f"\n✅ Done! Copied {copied_count} image-annotation pairs to {output_folder}")