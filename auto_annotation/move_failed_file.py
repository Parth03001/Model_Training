import shutil
import os

# #black rim black cap
# txt_file = "D:\\Training_Scripts\\auto_annotated_black_rim_black_cap\\review_needed.txt"
# source_folder = "D:\\Training_Scripts\\Wheel_Data\\AX7_OK"
# dest_folder = "D:\\Training_Scripts\\failed_annotation\\black_rim_black_cap"

# #black rim grey cap
# txt_file = "D:\\Training_Scripts\\auto_annotated_black_rim_grey_cap\\review_needed.txt"
# source_folder = "D:\\Training_Scripts\\Wheel_Data\\AX7_NOT_OK"
# dest_folder = "D:\\Training_Scripts\\failed_annotation\\black_rim_grey_cap"

# #grey rim grey cap
# txt_file = "D:\\Training_Scripts\\auto_annotated_grey_rim_grey_cap\\review_needed.txt"
# source_folder = "D:\\Training_Scripts\\Wheel_Data\\AX7L_OK"
# dest_folder = "D:\\Training_Scripts\\failed_annotation\\grey_rim_grey_cap"

#grey rim black cap
txt_file = "D:\\Training_Scripts\\auto_annotated_grey_rim_black_cap\\review_needed.txt"
source_folder = "D:\\Training_Scripts\\Wheel_Data\\AX7L_NOT_OK"
dest_folder = "D:\\Training_Scripts\\failed_annotation\\grey_rim_black_cap"

with open(txt_file, 'r') as f:
    image_names = [line.strip() for line in f if line.strip()]

print(f"Found {len(image_names)} images to move.")
for img_name in image_names:
    src = os.path.join(source_folder, img_name)
    dst = os.path.join(dest_folder, img_name)    
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied: {img_name}")
    else:
        print(f"Not found: {img_name}")

print("Done!")