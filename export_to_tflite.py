"""
Export trained EfficientNet-B0 model to TFLite for Flutter deployment.
 
Usage:
    python export_to_tflite.py --checkpoint wheel_model_output/best_model.pth --output wheel_efficientnet.tflite
"""
 
import torch
import torch.nn as nn
import argparse
import json
from pathlib import Path
 
 
def build_model(num_classes=4, dropout=0.3):
    # Match architecture from train_wheel_efficientnet.py
    from torchvision.models import efficientnet_v2_s
    model = efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model
 
 
def export_tflite(checkpoint_path, output_path, num_classes=4):
    try:
        import onnx
        import onnx2tf
        import shutil
        import os
    except ImportError:
        print("Install required packages:")
        print("  pip install onnx onnx2tf tensorflow")
        return
 
    # Load PyTorch model
    model = build_model(num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
 
    # Step 1: PyTorch -> ONNX
    # EfficientNetV2-S uses 384x384 input resolution
    onnx_path = output_path.replace('.tflite', '.onnx')
    dummy_input = torch.randn(1, 3, 384, 384)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=13,
        dynamic_axes=None
    )
    print(f"ONNX saved: {onnx_path}")
 
    # Step 2: ONNX -> TFLite (via onnx2tf)
    print("Converting ONNX to TFLite via onnx2tf...")
    output_folder = output_path.replace('.tflite', '_tflite_out')
    
    # Run onnx2tf conversion
    # It automatically saves tflite model in the output folder
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=output_folder,
        output_dynamic_range_quantized_tflite=False,
        output_integer_quantized_tflite=False,
        non_verbose=True
    )
    
    # Move the result to desired path
    # onnx2tf typically outputs 'model_float32.tflite' or similar
    # We look for the .tflite file in the output folder
    found = False
    for f in os.listdir(output_folder):
        if f.endswith('.tflite') and not f.endswith('_quant.tflite'):
            src = os.path.join(output_folder, f)
            shutil.move(src, output_path)
            found = True
            break
            
    if found:
        print(f"TFLite saved: {output_path}")
        # cleanup
        # shutil.rmtree(output_folder) # Optional: keep debug info or remove
    else:
        print(f"Error: TFLite model not found in {output_folder}")
 
 
def export_onnx_only(checkpoint_path, output_path, num_classes=4):
    """Simpler export: just ONNX. Convert to TFLite using external tools if needed."""
    model = build_model(num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
 
    onnx_path = output_path.replace('.tflite', '.onnx')
    dummy_input = torch.randn(1, 3, 384, 384)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=13,
    )
    print(f"ONNX saved: {onnx_path}")
    print("Convert to TFLite with: onnx2tf -i model.onnx -o tflite_output")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="wheel_model_output/best_model.pth")
    parser.add_argument("--output", default="wheel_efficientnet_AUG.tflite")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--onnx-only", action="store_true", help="Only export ONNX (skip TFLite)")
    args = parser.parse_args()
 
    if args.onnx_only:
        export_onnx_only(args.checkpoint, args.output, args.num_classes)
    else:
        export_tflite(args.checkpoint, args.output, args.num_classes)