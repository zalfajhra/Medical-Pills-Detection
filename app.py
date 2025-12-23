import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
from pathlib import Path
import yaml
import time

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="Deteksi Obat & Vitamin",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODEL DAN KONFIGURASI
# ============================================================================
@st.cache_resource
def load_model():
    """Load YOLOv11 model"""
    try:
        model_path = "models/best.pt"
        if not os.path.exists(model_path):
            st.error(f"❌ Model tidak ditemukan di: {model_path}")
            st.info("📥 Pastikan file 'best.pt' ada di folder 'models/'")
            return None
        
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_data
def load_class_names():
    """Load class names dari data.yaml"""
    try:
        with open('data.yaml', 'r') as f:
            data = yaml.safe_load(f)
            return data['names']
    except Exception as e:
        st.warning(f"⚠️ Tidak bisa load class names: {str(e)}")
        return None

# ============================================================================
# FUNGSI UTILITAS
# ============================================================================
def create_folders():
    """Buat folder yang dibutuhkan"""
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("snapshots", exist_ok=True)

def process_image(image, model, conf_threshold, iou_threshold):
    """Proses deteksi pada gambar"""
    try:
        img_array = np.array(image)
        results = model.predict(
            source=img_array,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            verbose=False
        )
        annotated_img = results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        return annotated_img, results[0]
    except Exception as e:
        st.error(f"❌ Error saat memproses gambar: {str(e)}")
        return None, None

def process_video_frames(video_path, model, conf_threshold, iou_threshold, progress_bar, status_text):
    """Proses video dan simpan frame-by-frame"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_dir = os.path.join("outputs", "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        frame_count = 0
        detection_summary = []
        frame_paths = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=640,
                verbose=False
            )
            
            annotated_frame = results[0].plot()
            frame_filename = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, annotated_frame)
            frame_paths.append(frame_filename)
            
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Memproses frame {frame_count}/{total_frames}")
            
            if len(results[0].boxes) > 0:
                detection_summary.append({
                    'frame': frame_count,
                    'detections': len(results[0].boxes)
                })
        
        cap.release()
        
        import imageio
        output_path = os.path.join("outputs", "detected_video.mp4")
        
        status_text.text("📹 Menyusun video...")
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')
        
        for frame_path in frame_paths:
            frame = imageio.imread(frame_path)
            writer.append_data(frame)
        
        writer.close()
        
        import shutil
        shutil.rmtree(frames_dir)
        
        return output_path, detection_summary
    
    except Exception as e:
        st.error(f"❌ Error saat memproses video: {str(e)}")
        return None, None

def display_detection_info(results, class_names):
    """Tampilkan informasi deteksi"""
    boxes = results.boxes
    
    if len(boxes) == 0:
        st.warning("⚠️ Tidak ada objek yang terdeteksi")
        return
    
    st.success(f"✅ Terdeteksi {len(boxes)} objek!")
    
    detection_data = []
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        if class_names and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = f"Class {cls_id}"
        
        detection_data.append({
            "No": i + 1,
            "Obat/Vitamin": class_name,
            "Confidence": f"{conf:.2%}"
        })
    
    st.dataframe(detection_data, use_container_width=False)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Deteksi", len(boxes))
    with col2:
        avg_conf = np.mean([float(box.conf[0]) for box in boxes])
        st.metric("Rata-rata Confidence", f"{avg_conf:.2%}")
    with col3:
        unique_classes = len(set([int(box.cls[0]) for box in boxes]))
        st.metric("Jenis Obat Berbeda", unique_classes)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    create_folders()
    
    st.title("💊 Aplikasi Deteksi Obat & Vitamin")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        
        with st.spinner("🔄 Loading model..."):
            model = load_model()
            class_names = load_class_names()
        
        if model is None:
            st.stop()
        
        st.success("✅ Model berhasil dimuat!")
        
        if class_names:
            st.info(f"📊 Jumlah kelas: {len(class_names)}")
            with st.expander("📋 Daftar Kelas"):
                for i, name in enumerate(class_names):
                    st.text(f"{i+1}. {name}")
        
        st.markdown("---")
        st.subheader("🎯 Pengaturan Deteksi")
        
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Semakin tinggi = lebih selektif"
        )
        
        iou_threshold = st.slider(
            "IOU Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Untuk menghilangkan duplikat deteksi"
        )
        
        st.markdown("---")
        st.markdown("### 📖 Cara Pakai:")
        st.markdown("""
        1. Pilih mode deteksi
        2. Upload file atau capture webcam
        3. Lihat hasil deteksi!
        """)
    
    tab1, tab2, tab3 = st.tabs(["📷 Deteksi Gambar", "🎬 Deteksi Video", "📹 Capture Webcam"])
    
    # TAB 1: DETEKSI GAMBAR
    with tab1:
        st.header("📷 Upload Gambar")
        
        uploaded_file = st.file_uploader(
            "Pilih gambar obat/vitamin",
            type=['jpg', 'jpeg', 'png'],
            help="Format: JPG, JPEG, PNG",
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🖼️ Gambar Asli")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            if st.button("🔍 Mulai Deteksi", type="primary", use_container_width=True, key="detect_image"):
                with st.spinner("🔄 Sedang memproses..."):
                    annotated_img, results = process_image(
                        image, model, conf_threshold, iou_threshold
                    )
                
                if annotated_img is not None:
                    with col2:
                        st.subheader("✅ Hasil Deteksi")
                        st.image(annotated_img, use_column_width=True)
                    
                    st.markdown("---")
                    st.subheader("📊 Informasi Deteksi")
                    display_detection_info(results, class_names)
                    
                    result_img = Image.fromarray(annotated_img)
                    buf = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    result_img.save(buf.name)
                    
                    with open(buf.name, 'rb') as f:
                        st.download_button(
                            label="📥 Download Hasil",
                            data=f,
                            file_name="detected_image.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )
    
    # TAB 2: DETEKSI VIDEO
    with tab2:
        st.header("🎬 Upload Video")
        
        uploaded_video = st.file_uploader(
            "Pilih video obat/vitamin",
            type=['mp4', 'avi', 'mov'],
            help="Format: MP4, AVI, MOV",
            key="video_uploader"
        )
        
        if uploaded_video is not None:
            temp_video_path = os.path.join("uploads", "temp_video.mp4")
            with open(temp_video_path, 'wb') as f:
                f.write(uploaded_video.read())
            
            st.subheader("🎥 Video Asli")
            st.video(temp_video_path)
            
            if st.button("🔍 Mulai Deteksi Video", type="primary", use_container_width=True, key="detect_video"):
                st.markdown("---")
                st.subheader("⏳ Memproses Video...")
                
                try:
                    import imageio_ffmpeg
                except ImportError:
                    st.info("📦 Menginstall imageio-ffmpeg...")
                    import subprocess
                    subprocess.check_call(['pip', 'install', 'imageio-ffmpeg'])
                    import imageio_ffmpeg
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                output_path, detection_summary = process_video_frames(
                    temp_video_path, model, conf_threshold, iou_threshold,
                    progress_bar, status_text
                )
                
                if output_path and os.path.exists(output_path):
                    status_text.text("✅ Proses selesai!")
                    
                    st.markdown("---")
                    st.subheader("✅ Hasil Deteksi")
                    st.video(output_path)
                    
                    if detection_summary:
                        st.markdown("---")
                        st.subheader("📊 Ringkasan Deteksi")
                        
                        total_detections = sum([d['detections'] for d in detection_summary])
                        frames_with_detection = len(detection_summary)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Deteksi", total_detections)
                        with col2:
                            st.metric("Frame dengan Deteksi", frames_with_detection)
                    
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Video Hasil",
                            data=f,
                            file_name="detected_video.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                else:
                    st.error("❌ Gagal memproses video")
    
    # TAB 3: WEBCAM CAPTURE (SIMPLE)
    with tab3:
        st.header("📹 Capture dari Webcam")
        
        st.info("""
        💡 **Cara Pakai:**
        1. Klik tombol "📸 Capture dari Webcam"
        2. Izinkan akses kamera ketika browser meminta
        3. Ambil foto dengan klik tombol di kamera
        4. Foto otomatis akan dideteksi!
        """)
        
        st.markdown("---")
        
        # Streamlit camera input (native, no dependencies!)
        camera_photo = st.camera_input("📸 Ambil foto dari webcam", key="webcam_capture")
        
        if camera_photo is not None:
            # Convert to PIL Image
            image = Image.open(camera_photo)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Foto dari Webcam")
                st.image(image, use_column_width=True)
            
            # Auto-detect
            with st.spinner("🔄 Mendeteksi objek..."):
                annotated_img, results = process_image(
                    image, model, conf_threshold, iou_threshold
                )
            
            if annotated_img is not None:
                with col2:
                    st.subheader("✅ Hasil Deteksi")
                    st.image(annotated_img, use_column_width=True)
                
                st.markdown("---")
                st.subheader("📊 Informasi Deteksi")
                display_detection_info(results, class_names)
                
                # Save and download options
                col_save, col_download = st.columns(2)
                
                with col_save:
                    if st.button("💾 Simpan Snapshot", use_container_width=True):
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        snapshot_path = os.path.join("snapshots", f"snapshot_{timestamp}.jpg")
                        result_img = Image.fromarray(annotated_img)
                        result_img.save(snapshot_path)
                        st.success(f"✅ Snapshot disimpan: {snapshot_path}")
                
                with col_download:
                    result_img = Image.fromarray(annotated_img)
                    buf = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    result_img.save(buf.name)
                    
                    with open(buf.name, 'rb') as f:
                        st.download_button(
                            label="📥 Download Hasil",
                            data=f,
                            file_name=f"webcam_detection_{time.strftime('%Y%m%d_%H%M%S')}.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )
        
        # Display saved snapshots
        st.markdown("---")
        st.subheader("📸 Snapshots Tersimpan")
        
        snapshot_files = sorted([f for f in os.listdir("snapshots") if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if len(snapshot_files) > 0:
            st.info(f"Total snapshots: {len(snapshot_files)}")
            
            # Display last 6 snapshots
            cols = st.columns(3)
            for idx, snapshot_file in enumerate(snapshot_files[-6:][::-1]):
                with cols[idx % 3]:
                    snapshot_path = os.path.join("snapshots", snapshot_file)
                    st.image(snapshot_path, caption=snapshot_file, use_column_width=True)
                    
                    # Delete button
                    if st.button(f"🗑️ Hapus", key=f"delete_{snapshot_file}"):
                        os.remove(snapshot_path)
                        st.rerun()
        else:
            st.info("Belum ada snapshot tersimpan. Capture foto dari webcam untuk mulai!")
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Tips untuk Hasil Terbaik:
        - ✅ Pencahayaan cukup terang
        - ✅ Tahan obat/vitamin dengan stabil
        - ✅ Jarak ideal: 20-40 cm dari kamera
        - ✅ Background polos untuk akurasi maksimal
        - ✅ Fokus kamera jelas, tidak blur
        """)

if __name__ == "__main__":
    main()