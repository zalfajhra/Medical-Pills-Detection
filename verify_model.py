"""
Script untuk memverifikasi class names dari model YOLOv11
Jalankan script ini untuk memastikan urutan class sudah benar
"""

from ultralytics import YOLO
import cv2

# Load model
MODEL_PATH = 'models/best.pt'
print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Print model information
print("\n" + "="*60)
print("MODEL INFORMATION")
print("="*60)

# Get class names
if hasattr(model, 'names'):
    class_names = model.names
    print(f"\nTotal classes: {len(class_names)}")
    print("\nClass mapping from model:")
    print("-"*60)
    for idx, name in class_names.items():
        print(f"  Class ID {idx}: {name}")
    print("-"*60)
else:
    print("Warning: Model doesn't have 'names' attribute")

# Test detection on a sample image (optional)
print("\n" + "="*60)
print("TESTING DETECTION")
print("="*60)

# You can test with a sample image
test_image = input("\nEnter path to test image (or press Enter to skip): ").strip()

if test_image:
    try:
        print(f"\nRunning detection on: {test_image}")
        results = model(test_image)
        
        print("\nDetection Results:")
        print("-"*60)
        
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = r.names[cls]
                    
                    print(f"  Detected: {class_name}")
                    print(f"    - Class ID: {cls}")
                    print(f"    - Confidence: {conf*100:.2f}%")
                    print()
            else:
                print("  No objects detected")
        
        print("-"*60)
        
        # Show annotated image
        annotated = results[0].plot()
        cv2.imshow('Detection Result', annotated)
        print("\nPress any key in the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during detection: {e}")
else:
    print("Skipping detection test")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
print("\nIf the class names above are correct, your Flask app should work properly.")
print("Make sure the class names in app.py match these names EXACTLY (including spaces and capitalization).")