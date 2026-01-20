import os

folder_path = "D:\\Training_Scripts\\AX7_OK_500"
base_name = "IMG_20260108"
start_num = 1

images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
images.sort()

for i, filename in enumerate(images, start=start_num):
    ext = os.path.splitext(filename)[1]
    new_name = f"{base_name}_img{i}{ext}"
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
    print(f"{filename} -> {new_name}")

print("Done!")