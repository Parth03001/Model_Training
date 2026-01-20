"""
Simple GUI Alternator Connector Inspector
No web server - Pure Python GUI using tkinter
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO
import threading

class ConnectorInspectorGUI:
    def __init__(self, root, yolo_path, classifier_path):
        self.root = root
        self.root.title("Alternator Connector Inspector")
        self.root.geometry("1400x900")
        
        # Load models
        self.load_models(yolo_path, classifier_path)
        
        # Setup GUI
        self.setup_gui()
        
    def load_models(self, yolo_path, classifier_path):
        """Load models"""
        self.status_text = "Loading models..."
        print("Loading YOLO detector...")
        self.yolo_model = YOLO(yolo_path)
        
        print("Loading classifier...")
        self.processor = AutoImageProcessor.from_pretrained(classifier_path)
        self.classifier_model = AutoModelForImageClassification.from_pretrained(classifier_path)
        self.classifier_model.eval()
        
        print("✓ Models loaded!")
        self.status_text = "Ready"
        
    def setup_gui(self):
        """Setup GUI elements"""
        
        # Title
        title = tk.Label(
            self.root,
            text="🔧 Alternator Connector Quality Inspector",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white",
            pady=15
        )
        title.pack(fill=tk.X)
        
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Input
        left_frame = tk.LabelFrame(
            main_frame,
            text="📤 Input Image",
            font=("Arial", 12, "bold"),
            padx=10,
            pady=10
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Image display (input)
        self.input_canvas = tk.Canvas(left_frame, width=500, height=400, bg="gray")
        self.input_canvas.pack()
        
        # Buttons
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(pady=10)
        
        self.upload_btn = tk.Button(
            btn_frame,
            text="📁 Upload Image",
            command=self.upload_image,
            font=("Arial", 12),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=10
        )
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        self.inspect_btn = tk.Button(
            btn_frame,
            text="🔍 Inspect",
            command=self.inspect_image,
            font=("Arial", 12),
            bg="#27ae60",
            fg="white",
            padx=20,
            pady=10,
            state=tk.DISABLED
        )
        self.inspect_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Output
        right_frame = tk.LabelFrame(
            main_frame,
            text="📊 Inspection Result",
            font=("Arial", 12, "bold"),
            padx=10,
            pady=10
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Image display (output)
        self.output_canvas = tk.Canvas(right_frame, width=500, height=400, bg="gray")
        self.output_canvas.pack()
        
        # Result text
        self.result_label = tk.Label(
            right_frame,
            text="",
            font=("Arial", 16, "bold"),
            pady=10
        )
        self.result_label.pack()
        
        # Details
        self.details_text = tk.Text(
            right_frame,
            height=6,
            font=("Arial", 10),
            wrap=tk.WORD
        )
        self.details_text.pack(fill=tk.BOTH, padx=10, pady=5)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Arial", 10)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Variables
        self.current_image = None
        self.image_path = None
        
    def upload_image(self):
        """Upload image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.image_path = file_path
            self.load_and_display_image(file_path)
            self.inspect_btn.config(state=tk.NORMAL)
            self.status_bar.config(text=f"Loaded: {file_path}")
            
    def load_and_display_image(self, path):
        """Load and display image"""
        image = Image.open(path).convert('RGB')
        self.current_image = image
        
        # Resize for display
        display_image = image.copy()
        display_image.thumbnail((500, 400))
        
        photo = ImageTk.PhotoImage(display_image)
        self.input_canvas.create_image(250, 200, image=photo, anchor=tk.CENTER)
        self.input_canvas.image = photo
        
    def inspect_image(self):
        """Inspect image in separate thread"""
        self.status_bar.config(text="Inspecting...")
        self.inspect_btn.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self.run_inspection)
        thread.start()
        
    def run_inspection(self):
        """Run inspection"""
        try:
            # Stage 1: YOLO Detection
            results = self.yolo_model(self.current_image, conf=0.25, verbose=False)
            
            if len(results[0].boxes) == 0:
                self.root.after(0, lambda: self.show_result(
                    None, "No connector detected", 0, 0
                ))
                return
            
            # Get detection
            boxes = results[0].boxes
            best_box = boxes[0]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            detection_conf = float(best_box.conf[0])
            
            # Crop
            img_array = np.array(self.current_image)
            h, w = img_array.shape[:2]
            
            padding = 0.1
            pad_w = int((x2 - x1) * padding)
            pad_h = int((y2 - y1) * padding)
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)
            
            cropped = img_array[y1:y2, x1:x2]
            cropped_pil = Image.fromarray(cropped)
            
            # Stage 2: Classification
            inputs = self.processor(images=cropped_pil, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.classifier_model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = probabilities.argmax().item()
                confidence = probabilities[0][predicted_class].item() * 100
            
            labels = ["OK", "NOT_OK"]  
            prediction = labels[predicted_class]
            
            # Create visualization
            vis_image = self.current_image.copy()
            draw = ImageDraw.Draw(vis_image)
            
            color = "green" if prediction == "OK" else "red"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=6)
            
            # Show result
            self.root.after(0, lambda: self.show_result(
                vis_image, prediction, confidence, detection_conf * 100
            ))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.status_bar.config(text="Error occurred"))
        
        finally:
            self.root.after(0, lambda: self.inspect_btn.config(state=tk.NORMAL))
            
    def show_result(self, vis_image, prediction, confidence, detection_conf):
        """Show inspection result"""
        
        if vis_image is None:
            self.result_label.config(
                text="❌ No Connector Detected",
                fg="red"
            )
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(1.0, "No connector found in the image.\nPlease try another image.")
            self.status_bar.config(text="No detection")
            return
        
        # Display result image
        display_image = vis_image.copy()
        display_image.thumbnail((500, 400))
        photo = ImageTk.PhotoImage(display_image)
        self.output_canvas.create_image(250, 200, image=photo, anchor=tk.CENTER)
        self.output_canvas.image = photo
        
        # Update result text
        if prediction == "OK":
            self.result_label.config(
                text=f"✅ Connector OK ({confidence:.1f}%)",
                fg="green"
            )
            description = "The alternator connector is properly connected and securely attached."
        else:
            self.result_label.config(
                text=f"❌ Connector NOT OK ({confidence:.1f}%)",
                fg="red"
            )
            description = "⚠️ Issue detected! The connector appears disconnected or improperly seated."
        
        # Update details
        details = f"""
🎯 Final Result: {prediction}
📊 Classification Confidence: {confidence:.2f}%
🔍 Detection Confidence: {detection_conf:.2f}%

📝 Description:
{description}
        """
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, details)
        
        self.status_bar.config(text=f"Inspection complete: {prediction}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GUI Connector Inspector")
    parser.add_argument('--yolo-model', type=str,
                       default='connector_detector/yolov8_connector/weights/best.pt')
    parser.add_argument('--classifier-model', type=str,
                       default='best_model_hf')
    
    args = parser.parse_args()
    
    root = tk.Tk()
    app = ConnectorInspectorGUI(root, args.yolo_model, args.classifier_model)
    root.mainloop()

if __name__ == "__main__":
    main()