import os

label_folder = r"C:\Users\50014665\Downloads\data\ok condition"  # Your XML folder


# Find conflicts
xml_txt_files = [f for f in os.listdir(label_folder) if f.endswith('.xml.txt')]
txt_files = [f for f in os.listdir(label_folder) if f.endswith('.txt') and not f.endswith('.xml.txt')]

conflicts = []
for xml_file in xml_txt_files:
    target_name = xml_file.replace('.xml.txt', '.txt')
    if target_name in txt_files:
        conflicts.append(xml_file)

print(f"Found {len(conflicts)} files with conflicts:")
for f in conflicts:
    print(f"  - {f} (conflicts with {f.replace('.xml.txt', '.txt')})")

# Option 1: Delete the .xml.txt duplicates (keeping .txt)
print("\n🗑️ Deleting .xml.txt duplicates...")
for xml_file in conflicts:
    file_path = os.path.join(label_folder, xml_file)
    os.remove(file_path)
    print(f"Deleted: {xml_file}")

# Option 2: Rename only non-conflicting files
safe_to_rename = [f for f in xml_txt_files if f not in conflicts]
print(f"\n✅ Renaming {len(safe_to_rename)} non-conflicting files...")
for filename in safe_to_rename:
    old_path = os.path.join(label_folder, filename)
    new_filename = filename.replace('.xml.txt', '.txt')
    new_path = os.path.join(label_folder, new_filename)
    os.rename(old_path, new_path)
    print(f"Renamed: {filename}")