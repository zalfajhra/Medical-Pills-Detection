from flask import Flask, render_template, request, jsonify, Response, send_from_directory, send_file
from ultralytics import YOLO
import cv2
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np
import base64

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# Create folders if not exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Load YOLO model
MODEL_PATH = 'models/best.pt'
model = YOLO(MODEL_PATH)

# Get class names directly from the model
# Ini akan mengambil nama class yang sebenarnya dari model yang sudah di-train
CLASS_NAMES = list(model.names.values()) if hasattr(model, 'names') else [
    'Alaxan', 'Biogesic', 'Decolgen', 'DayZinc', 'Medicol',
    'Kremil S', 'Neozep', 'Fishoil', 'Bioflu', 'Bactidol'
]

# Print class names for debugging
print("="*50)
print("Loaded model class names:")
for idx, name in enumerate(CLASS_NAMES):
    print(f"  Class {idx}: {name}")
print("="*50)

# Global variable for webcam
camera = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_image(image_path):
    """Detect objects in image"""
    # Read original image
    original_img = cv2.imread(image_path)
    
    # Run detection
    results = model(image_path)
    
    # Get annotated image
    annotated = results[0].plot()
    
    # Save both images
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    original_filename = f'original_{timestamp}.jpg'
    detected_filename = f'detected_{timestamp}.jpg'
    
    original_path = os.path.join(app.config['OUTPUT_FOLDER'], original_filename)
    detected_path = os.path.join(app.config['OUTPUT_FOLDER'], detected_filename)
    
    cv2.imwrite(original_path, original_img)
    cv2.imwrite(detected_path, annotated)
    
    # Extract detection info - menggunakan dictionary untuk menghindari duplikat
    unique_detections = {}
    
    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Get class name dari model, bukan dari list manual
                class_name = r.names[cls] if cls in r.names else CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f'Class_{cls}'
                
                # Simpan hanya confidence tertinggi untuk setiap class
                if class_name not in unique_detections or conf > unique_detections[class_name]:
                    unique_detections[class_name] = conf
    
    # Convert ke list format
    detections = []
    for class_name, conf in unique_detections.items():
        detections.append({
            'class': class_name,
            'confidence': round(conf * 100, 2)
        })
    
    # Sort by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return original_filename, detected_filename, detections

def detect_video(video_path):
    """Detect objects in video with H.264 codec for browser compatibility"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default fallback
    
    # Output video with H.264 codec for browser compatibility
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'detected_video_{timestamp}.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    # Use H.264 codec (avc1) which is widely supported by browsers
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    # Dictionary to track unique detections per class
    unique_detections = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect every frame
        results = model(frame, verbose=False)
        annotated = results[0].plot()
        
        # Extract detections from current frame
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Get class name dari model
                    class_name = r.names[cls] if cls in r.names else CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f'Class_{cls}'
                    
                    # Initialize if first time seeing this class
                    if class_name not in unique_detections:
                        unique_detections[class_name] = {
                            'max_confidence': conf,
                            'total_detections': 0
                        }
                    
                    # Update statistics
                    unique_detections[class_name]['total_detections'] += 1
                    if conf > unique_detections[class_name]['max_confidence']:
                        unique_detections[class_name]['max_confidence'] = conf
        
        out.write(annotated)
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Format detections for display - only include classes that were actually detected
    total_detections = []
    for class_name, stats in unique_detections.items():
        total_detections.append({
            'class': class_name,
            'confidence': round(stats['max_confidence'] * 100, 2),
            'count': stats['total_detections']
        })
    
    # Sort by confidence (highest first)
    total_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return output_filename, total_detections, frame_count

def generate_frames():
    """Generate frames for webcam streaming"""
    global camera
    
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Detect objects
        results = model(frame, verbose=False)
        annotated = results[0].plot()
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect
        original_filename, detected_filename, detections = detect_image(filepath)
        
        return jsonify({
            'success': True,
            'original_image': f'/outputs/{original_filename}',
            'output_image': f'/outputs/{detected_filename}',
            'detections': detections,
            'total_detected': len(detections),
            'download_url': f'/download/{detected_filename}'
        })
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect
        output_filename, detections, frame_count = detect_video(filepath)
        
        return jsonify({
            'success': True,
            'output_video': f'/outputs/{output_filename}',
            'detections': detections,
            'total_detected': len(detections),
            'frames_processed': frame_count,
            'download_url': f'/download/{output_filename}'
        })
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'success': True})

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    """Download detection result"""
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)