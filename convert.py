import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os

# CHANGE THIS to your actual model path
model_path = "C:\\Users\\50014665\\Image_Classification\\connector_inspection\\yolo11s_single_stage\\weights\\best.onnx"  

print("=" * 50)
print("ONNX to TFLite Converter")
print("=" * 50)

# Check if model exists
if not os.path.exists(model_path):
    print(f"❌ Error: Model not found at {model_path}")
    print("Please update 'model_path' in the script")
    exit(1)

# Load ONNX model
print(f"\n1. Loading ONNX model from {model_path}...")
onnx_model = onnx.load(model_path)
print("   ✓ ONNX model loaded")

# Convert to TensorFlow
print("\n2. Converting ONNX → TensorFlow...")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("saved_model")
print("   ✓ TensorFlow model created")

# Convert to TFLite
print("\n3. Converting TensorFlow → TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantization for smaller size
tflite_model = converter.convert()
print("   ✓ TFLite model created")

# Save
output_path = "classifier.tflite"
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print("\n" + "=" * 50)
print("✓ SUCCESS!")
print("=" * 50)
print(f"Model size: {len(tflite_model)/1024:.2f} KB")
print(f"Saved as: {os.path.abspath(output_path)}")
print("\nYou can now use this model in your Flutter app!")