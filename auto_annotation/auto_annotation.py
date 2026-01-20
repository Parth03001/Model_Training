from ultralytics import YOLO
from pathlib import Path
import shutil
import cv2

# ============================================
# FINAL CLASS MAPPING (MASTER REFERENCE)
# ============================================
# rim_black → 0
# cap_black → 1
# rim_grey  → 2
# cap_grey  → 3

# ============================================
# CORRECT CONFIGURATION - USE THIS!
# ============================================

# ---------- COMBINATION 1: Black Rim + Black Cap ----------
all_images_dir = 'D:\\Training_Scripts\\Wheel_Data\\AX7_OK'
output_dir = 'D:\\Training_Scripts\\Training_Scripts\\auto_annotated_black_rim_black_cap'
class_remap = {
    0: 0,  # rim_black → rim_black (no change)
    1: 1   # cap_black → cap_black (no change)
}
class_names = {0: 'rim_black', 1: 'cap_black'}
colors = {0: (0, 255, 0), 1: (0, 255, 255)}

# ---------- COMBINATION 2: Black Rim + Grey Cap ----------
# all_images_dir = 'D:\\Training_Scripts\\Wheel_Data\\AX7_NOT_OK'
# output_dir = 'D:\\Training_Scripts\\Training_Scripts\\auto_annotated_black_rim_grey_cap'
# class_remap = {
#     0: 0,  # rim_black → rim_black (no change)
#     1: 3   # cap_black → cap_grey (CHANGE COLOR)
# }
# class_names = {0: 'rim_black', 3: 'cap_grey'}
# colors = {0: (0, 255, 0), 3: (0, 255, 255)}

# ---------- COMBINATION 3: Grey Rim + Grey Cap ----------
# all_images_dir = 'D:\\Training_Scripts\\Wheel_Data\\AX7L_OK'
# output_dir = 'D:\\Training_Scripts\\Training_Scripts\\auto_annotated_grey_rim_grey_cap'
# class_remap = {
#     0: 2,  # rim_black → rim_grey (CHANGE COLOR)
#     1: 3   # cap_black → cap_grey (CHANGE COLOR)
# }
# class_names = {2: 'rim_grey', 3: 'cap_grey'}
# colors = {2: (0, 255, 0), 3: (0, 255, 255)}

# # ---------- COMBINATION 4: Grey Rim + Black Cap ----------
# all_images_dir = 'D:\\Training_Scripts\\Wheel_Data\\AX7L_NOT_OK'
# output_dir = 'D:\\Training_Scripts\\Training_Scripts\\auto_annotated_grey_rim_black_cap'
# class_remap = {
#     0: 2,  # rim_black → rim_grey (CHANGE COLOR)
#     1: 1   # cap_black → cap_black (no change)
# }
# class_names = {2: 'rim_grey', 1: 'cap_black'}
# colors = {2: (0, 255, 0), 1: (0, 255, 255)}

# Model path (same for all)
model_path = r'D:\Training_Scripts\Scripts\auto_annotation\runs\detect\runs\bootstrap\wheel_model\weights\best.pt'

# ============================================
# AUTO-ANNOTATION WITH CLASS REMAPPING
# ============================================

print("🚀 Starting auto-annotation with class remapping...\n")
print(f"Combination: {list(class_names.values())}")
print(f"Class mapping: {class_remap}\n")

model = YOLO(model_path)

output_path = Path(output_dir)
(output_path / 'images').mkdir(parents=True, exist_ok=True)
(output_path / 'labels').mkdir(parents=True, exist_ok=True)
(output_path / 'visualizations' / 'good_detections').mkdir(parents=True, exist_ok=True)
(output_path / 'visualizations' / 'low_confidence').mkdir(parents=True, exist_ok=True)
(output_path / 'visualizations' / 'no_detections').mkdir(parents=True, exist_ok=True)
(output_path / 'visualizations' / 'multiple_detections').mkdir(parents=True, exist_ok=True)

all_images_set = set()
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
    all_images_set.update(Path(all_images_dir).glob(ext))

all_images = list(all_images_set)

print(f"Found {len(all_images)} unique images\n")

low_confidence = []
no_detection = []
good_detection = []
multiple_detections = []
skipped = []

for idx, img_path in enumerate(all_images):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Skipped: {img_path.name}")
            skipped.append(img_path.name)
            continue
        
        results = model(img, conf=0.25, verbose=False)
        
        img_vis = img.copy()
        label_file = output_path / 'labels' / f"{img_path.stem}.txt"
        
        # Check if no detections
        if len(results[0].boxes) == 0:
            shutil.copy(img_path, output_path / 'images' / img_path.name)
            no_detection.append(img_path.name)
            label_file.touch()
            
            cv2.putText(img_vis, "NO DETECTION", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imwrite(str(output_path / 'visualizations' / 'no_detections' / img_path.name), img_vis)
            
            if (idx + 1) % 50 == 0:
                print(f"Processed: {idx + 1}/{len(all_images)}")
            continue
        
        # Check if more than 2 detections - SKIP THIS IMAGE
        if len(results[0].boxes) > 2:
            multiple_detections.append(img_path.name)
            
            # Draw all boxes for visualization
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                old_cls = int(box.cls[0])
                new_cls = class_remap.get(old_cls, old_cls)
                conf = float(box.conf[0])
                
                color = colors[new_cls]
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 3)
                
                label_text = f"{class_names[new_cls]} {conf:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(img_vis, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 255), -1)
                cv2.putText(img_vis, label_text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add warning text
            cv2.putText(img_vis, f"TOO MANY DETECTIONS: {len(results[0].boxes)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imwrite(str(output_path / 'visualizations' / 'multiple_detections' / img_path.name), img_vis)
            
            if (idx + 1) % 50 == 0:
                print(f"Processed: {idx + 1}/{len(all_images)}")
            continue
        
        # Valid detections (1 or 2 boxes)
        shutil.copy(img_path, output_path / 'images' / img_path.name)
        
        max_conf = 0
        with open(label_file, 'w') as f:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x, y, w, h = box.xywhn[0].tolist()
                
                old_cls = int(box.cls[0])
                new_cls = class_remap.get(old_cls, old_cls)
                
                conf = float(box.conf[0])
                max_conf = max(max_conf, conf)
                
                f.write(f"{new_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                
                color = colors[new_cls]
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 3)
                
                label_text = f"{class_names[new_cls]} {conf:.2f}"
                label_bg_color = (0, 255, 0) if conf > 0.7 else (0, 165, 255) if conf > 0.5 else (0, 0, 255)
                
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(img_vis, (x1, y1 - text_height - 10), (x1 + text_width, y1), label_bg_color, -1)
                cv2.putText(img_vis, label_text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if max_conf < 0.7:
            low_confidence.append(img_path.name)
            cv2.imwrite(str(output_path / 'visualizations' / 'low_confidence' / img_path.name), img_vis)
        else:
            good_detection.append(img_path.name)
            cv2.imwrite(str(output_path / 'visualizations' / 'good_detections' / img_path.name), img_vis)
        
        if (idx + 1) % 50 == 0:
            print(f"Processed: {idx + 1}/{len(all_images)}")
            
    except Exception as e:
        print(f"❌ Error: {img_path.name} - {str(e)}")
        skipped.append(img_path.name)
        continue

print(f"\n{'='*60}")
print(f"✅ COMPLETE!")
print(f"{'='*60}")
print(f"\nResults:")
print(f"   Total: {len(all_images)}")
print(f"   Good (>0.7): {len(good_detection)}")
print(f"   Low conf: {len(low_confidence)}")
print(f"   No detect: {len(no_detection)}")
print(f"   Multiple detect (>2): {len(multiple_detections)}")
print(f"   Skipped: {len(skipped)}")

with open(output_path / 'review_needed.txt', 'w') as f:
    f.write("# HIGH PRIORITY - Multiple detections (>2 boxes)\n")
    f.write("\n".join(multiple_detections))
    f.write("\n\n# HIGH PRIORITY - No detections\n")
    f.write("\n".join(no_detection))
    f.write("\n\n# MEDIUM PRIORITY - Low confidence\n")
    f.write("\n".join(low_confidence))

if skipped:
    with open(output_path / 'skipped_files.txt', 'w') as f:
        f.write("# Skipped files\n")
        f.write("\n".join(skipped))

print(f"\n✅ Saved to: {output_dir}/")
print(f"🟢 GREEN box = rim")
print(f"🟡 YELLOW box = cap")
print(f"\n⚠️ Images with >2 detections saved to: visualizations/multiple_detections/")