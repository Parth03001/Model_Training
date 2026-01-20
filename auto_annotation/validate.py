# ============================================
# VALIDATE BEST MODEL
# ============================================
from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':
    best_model_path = Path('D:\\Training_Scripts\\Scripts\\auto_annotation\\wheel_training\\yolo11m_4class_aggressive_aug\\weights\\best.pt')
    print(f"\n📁 Best model: {best_model_path}")

    best_model = YOLO(best_model_path)

    print("\n📊 Validating model...")
    metrics = best_model.val()

    print(f"\n{'='*60}")
    print(f"📈 OVERALL PERFORMANCE")
    print(f"{'='*60}")
    print(f"mAP@0.5:      {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"Precision:    {metrics.box.mp:.3f}")
    print(f"Recall:       {metrics.box.mr:.3f}")

    print(f"\n{'='*60}")
    print(f"📊 PER-CLASS PERFORMANCE")
    print(f"{'='*60}")

    class_names = ['rim_black', 'cap_black', 'rim_grey', 'cap_grey']
    for i, name in enumerate(class_names):
        print(f"\n{name} (class {i}):")
        print(f"  Precision: {metrics.box.class_result(i)[0]:.3f}")
        print(f"  Recall:    {metrics.box.class_result(i)[1]:.3f}")
        print(f"  mAP@0.5:   {metrics.box.class_result(i)[2]:.3f}")

    print(f"\n{'='*60}")
    print(f"🎯 NEXT STEPS")
    print(f"{'='*60}")
    print(f"\n1. Review training plots:")
    print(f"   📁 runs/final_training/yolo11m_4class_aggressive_aug/")
    print(f"\n2. Check results.csv for metrics history")
    print(f"\n3. Test inference on new images")
    print(f"\n4. Export for deployment (ONNX, TensorRT)")

    # ============================================
    # EXPORT FOR PRODUCTION
    # ============================================

    print(f"\n{'='*60}")
    print(f"📦 EXPORTING MODEL")
    print(f"{'='*60}")

    # ONNX format (universal - works everywhere)
    print("\n⏳ Exporting to ONNX...")
    best_model.export(format='onnx', simplify=True)
    print("✅ ONNX exported: best.onnx")

    # TorchScript (PyTorch native)
    print("\n⏳ Exporting to TorchScript...")
    best_model.export(format='torchscript')
    print("✅ TorchScript exported: best.torchscript")
    
    # TF Lite (PyTorch native)
    print("\n⏳ Exporting to TF Lite...")
    best_model.export(format='tflite')
    print("✅ TF Lite exported: best.tflite")
    

    print(f"\n{'='*60}")
    print(f"✅ ALL DONE!")
    print(f"{'='*60}")
    print(f"\nBest model: {best_model_path}")
    print(f"Training logs: runs/final_training/yolo11m_4class_aggressive_aug/")