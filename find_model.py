# find_model.py
import os

for root, dirs, files in os.walk('.'):
    for f in files:
        if 'config.json' in f or '.safetensors' in f:
            print(f"Found: {os.path.join(root, f)}")