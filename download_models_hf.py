import os
import shutil
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: 'huggingface_hub' library not found.")
    print("Please install it using: pip install huggingface_hub")
    sys.exit(1)

# 1. Setup Directories
base_dir = "datavision_hf_models"
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.makedirs(base_dir, exist_ok=True)

print(f"🚀 Starting full repository downloads to: {os.path.abspath(base_dir)}")

# ---------------------------------------------------------
# Define the Repositories to Download
# ---------------------------------------------------------
repos_to_download = [
    # Repo ID                                   # Local Folder Name
    ("IDEA-Research/grounding-dino-base",       "grounding-dino-base"),
    ("openai/clip-vit-large-patch14",           "clip-vit-large-patch14"),
    ("facebook/sam2-hiera-large",               "sam2-hiera-large"),
]

# ---------------------------------------------------------
# Download Loop
# ---------------------------------------------------------
for repo_id, folder_name in repos_to_download:
    print(f"\n⬇️ Downloading full repo: {repo_id}...")
    local_path = os.path.join(base_dir, folder_name)
    
    try:
        # Download EVERYTHING (no filtering)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_path,
            ignore_patterns=None, 
        )
        print(f"✅ Finished: {folder_name}")
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")

# ---------------------------------------------------------
# Zip Everything
# ---------------------------------------------------------
zip_filename = "datavision_hf_models"
print(f"\n📦 Zipping all repositories (this will take a while)...")
try:
    shutil.make_archive(zip_filename, 'zip', base_dir)
    file_size_gb = os.path.getsize(f"{zip_filename}.zip") / (1024**3)
    print(f"\n✅ SUCCESS!")
    print(f"Zip File: {os.path.abspath(zip_filename + '.zip')}")
    print(f"Total Size: {file_size_gb:.2f} GB")
except Exception as e:
    print(f"❌ Zipping failed: {e}")
