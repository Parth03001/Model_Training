import os

image_folder = r"C:\Users\50014665\Downloads\data\ok condition"  # Your image folder
label_folder = r"C:\Users\50014665\Downloads\data\ok condition"  # Your XML folder

print("Images:", len([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]))
print("Labels:", len([f for f in os.listdir(label_folder) if f.endswith('.txt')]))