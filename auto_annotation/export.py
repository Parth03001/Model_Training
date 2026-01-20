import subprocess
import os

# Your ONNX model path
onnx_path = "D:\\Training_Scripts\\Scripts\\auto_annotation\\wheel_training\\yolo11m_4class_aggressive_aug\\weights\\best.onnx"
output_folder = "D:\\Training_Scripts\\Scripts\\auto_annotation\\wheel_training\\tflite_output"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Convert using onnx2tf
cmd = f"onnx2tf -i {onnx_path} -o {output_folder}"
subprocess.run(cmd, shell=True)

print(f"✓ TFLite models saved in: {output_folder}")