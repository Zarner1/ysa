import os
import glob
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# Note: importing `ultralytics` at module import time on Streamlit Cloud
# can fail if opencv or other binaries are not yet available. We perform
# a lazy import inside `load_yolo_model()` below to avoid startup import errors.
import time


# Import RCNN utilities
from main_rcnn import build_model, infer_classes_from_xmls, DATA_ROOT as RCNN_DATA_ROOT


PROJECT_ROOT = Path(__file__).parent
RUNS_DIR = PROJECT_ROOT / "runs"
# DATA_ROOT: önce ortam değişkenini kontrol et; yoksa proje içindeki `data` dizinini kullan
DATA_ROOT = Path(os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data")))
if not DATA_ROOT.exists():
    # fallback: proje kökü (örnek resimleriniz burada olabilir)
    DATA_ROOT = PROJECT_ROOT

# Test görüntü dizini: öncelikle `data/test/images`, yoksa proje kökü
TEST_IMG_DIR = DATA_ROOT / "test" / "images"
if not TEST_IMG_DIR.exists():
    TEST_IMG_DIR = PROJECT_ROOT

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def _find_first_existing(*paths, pattern=None):
    """Verilen yollar arasında ilk mevcut olanı döndürür. Eğer pattern verilirse
    proje kökünde bu paterne uyan ilk dosyayı arar.
    """
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            return str(p)
    if pattern:
        candidates = list(PROJECT_ROOT.glob(pattern))
        if candidates:
            return str(candidates[0])
    # Son çare: proje kökünde best*.pt veya *.pth araması
    for pat in ("best*.pt", "*.pt", "*.pth"):
        candidates = list(PROJECT_ROOT.glob(pat))
        if candidates:
            return str(candidates[0])
    # runs altındaki ağırlıkları da kontrol et
    candidates = list(RUNS_DIR.rglob("best*.pt")) if RUNS_DIR.exists() else []
    if candidates:
        return str(candidates[0])
    return None


@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    try:
        # Lazy import to avoid import-time failures on deployment platforms
        from ultralytics import YOLO
    except Exception as e:
        # Surface a friendly Streamlit error instead of crashing at import time
        st.error("`ultralytics` import failed. Check logs and requirements (opencv).")
        raise

    return YOLO(weights_path)


@st.cache_resource(show_spinner=False)
def load_rcnn_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sınıfları belirle
    annotations_path = os.path.join(RCNN_DATA_ROOT, "Annotations")
    classes = infer_classes_from_xmls(annotations_path)
    num_classes = len(classes) + 1  # +1 for background
    
    # Model oluştur ve checkpoint yükle
    model = build_model(num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, classes, device


def infer_image_yolo(model, img_path: Path, conf: float, iou: float, imgsz: int):
    # END-to-END: predict + postprocess(plot) dahil
    t0 = time.perf_counter()

    results = model.predict(
        source=str(img_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )

    # render overlay (postprocess + çizim)
    arr = results[0].plot()   # BGR numpy array
    arr = arr[:, :, ::-1]     # BGR -> RGB
    out_img = Image.fromarray(arr)

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    return out_img, latency_ms


def infer_image_rcnn(model, classes, device, img_path: Path, conf: float):
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import io
    import time

    # END-to-END: image load + tensor + inference + görselleştirme + buffer dahil
    t0 = time.perf_counter()

    # 1) Load + preprocess
    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = T.ToTensor()(img_pil).to(device)

    # 2) Inference
    with torch.no_grad():
        prediction = model([img_tensor])[0]

    # 3) Postprocess (threshold)
    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()

    mask = scores >= conf
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    # 4) Visualization
    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(img_pil)

    colors = ["red", "blue", "green", "yellow", "cyan", "magenta", "orange", "purple"]
    class_names = ["__background__"] + classes

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        color = colors[int(label) % len(colors)]
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=3, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        class_name = class_names[int(label)] if int(label) < len(class_names) else f"Class_{label}"
        ax.text(
            x1, max(0, y1 - 10),
            f"{class_name}: {score:.2f}",
            bbox=dict(facecolor=color, alpha=0.8, pad=5),
            fontsize=14, fontweight="bold", color="white"
        )

    ax.axis("off")
    plt.tight_layout()

    # 5) Figure -> PIL
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    out_img = Image.open(buf).copy()  # copy() önemli: buffer kapanınca bozulmasın
    buf.close()
    plt.close(fig)

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    return out_img, latency_ms


def infer_video(model: YOLO, video_path: Path, conf: float, iou: float, imgsz: int):
    # Process video and return 5 sample frames with detections from different segments
    results = model.predict(
        source=str(video_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=False,  # Don't save video, we'll extract frames
        stream=True,  # Generator for memory efficiency
        verbose=False,
    )
    
    # Collect 5 sample frames from different 100-frame segments
    sample_frames = []
    frame_count = 0
    total_detections = 0
    current_segment = 0
    segment_size = 100
    max_segments = 5
    
    for result in results:
        frame_count += 1
        num_boxes = len(result.boxes) if result.boxes is not None else 0
        total_detections += num_boxes
        
        # Determine which segment we're in (0-100, 100-200, 200-300, etc.)
        segment = (frame_count - 1) // segment_size
        
        # If we've moved to a new segment and haven't collected a frame from it yet
        if segment < max_segments and segment >= len(sample_frames):
            # Save first frame with detections in this segment
            if num_boxes > 0:
                arr = result.plot()  # BGR numpy array with bounding boxes
                arr = arr[:, :, ::-1]  # BGR -> RGB
                sample_frames.append(Image.fromarray(arr))
        
        # Stop after processing 500 frames (5 segments of 100)
        if frame_count >= max_segments * segment_size:
            break
    
    return sample_frames, frame_count, total_detections


def main():
    st.set_page_config(page_title="X-Ray Baggage Detection Demo", layout="centered", initial_sidebar_state="expanded")
    
    # Modern, profesyonel tasarım - Soft pastel tema
    st.markdown("""
        <style>
        /* Ana arka plan - soft pastel gradient */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #f0f4f8 0%, #e8eef5 100%);
        }
        
        /* Header */
        [data-testid="stHeader"] {
            background: rgba(240, 244, 248, 0.95);
            backdrop-filter: blur(10px);
        }
        
        /* Ana içerik alanı - sidebar'a yapışık */
        .main .block-container {
            margin-left: 0 !important;
            padding-left: 1rem !important;
            padding-right: 2rem !important;
            padding-top: 2rem;
            max-width: 100% !important;
        }
        
        .main {
            padding-left: 0 !important;
        }
        
        section[data-testid="stSidebar"] + section {
            padding-left: 0 !important;
        }
        
        /* Başlık stil - soft blue */
        h1 {
            color: #5a7fa5 !important;
            text-align: center;
            font-weight: 700 !important;
            font-size: 2.5rem !important;
            margin-bottom: 2rem !important;
            text-shadow: 0 2px 8px rgba(90, 127, 165, 0.15);
        }
        
        h2, h3 {
            color: #6b8fb3 !important;
            font-weight: 600 !important;
        }
        
        /* Sidebar stil - soft gradient */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f5f7fa 0%, #edf1f7 100%);
            border-right: 1px solid rgba(90, 127, 165, 0.1);
        }
        
        [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .st-emotion-cache-1cypcdb {
            color: #4a5568 !important;
        }
        
        /* Tab stil - soft colors */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(255, 255, 255, 0.6);
            padding: 0.5rem;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(157, 192, 220, 0.15);
            color: #5a7fa5;
            border-radius: 10px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            border: 1px solid rgba(157, 192, 220, 0.25);
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #9dc0dc 0%, #7ba7c9 100%) !important;
            color: white !important;
            box-shadow: 0 4px 15px rgba(157, 192, 220, 0.35);
        }
        
        /* Butonlar - soft blue gradient */
        .stButton button {
            background: linear-gradient(135deg, #9dc0dc 0%, #7ba7c9 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 2rem;
            font-weight: 500;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(157, 192, 220, 0.25);
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(157, 192, 220, 0.35);
            background: linear-gradient(135deg, #8bb4d3 0%, #6a98bd 100%);
        }
        
        /* Info, Success, Warning kutuları - soft */
        .stAlert {
            background-color: rgba(255, 255, 255, 0.85) !important;
            color: #4a5568 !important;
            border-radius: 12px !important;
            border-left: 4px solid #9dc0dc !important;
            backdrop-filter: blur(10px);
        }
        
        /* Görüntü caption'ları - soft borders */
        .stImage > div {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(90, 127, 165, 0.12);
            border: 2px solid rgba(157, 192, 220, 0.25);
            transition: all 0.3s ease;
        }
        
        .stImage > div:hover {
            transform: scale(1.01);
            box-shadow: 0 12px 32px rgba(157, 192, 220, 0.2);
            border-color: rgba(157, 192, 220, 0.4);
        }
        
        .stImage img {
            min-height: 400px;
            object-fit: contain;
        }
        
        /* File uploader - soft */
        [data-testid="stFileUploader"] {
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 12px;
            padding: 1rem;
            border: 2px dashed rgba(157, 192, 220, 0.35);
        }
        
        /* Checkbox */
        .stCheckbox {
            color: #4a5568 !important;
        }
        
        /* Selectbox ve diğer input'lar */
        .stSelectbox label, .stSlider label {
            color: #4a5568 !important;
            font-weight: 500 !important;
        }
        
        /* Metin renkleri - soft gray */
        p, label, span {
            color: #5a6a7a !important;
        }
        
        /* Spinner - soft blue */
        .stSpinner > div {
            border-top-color: #9dc0dc !important;
        }
        
        /* Subheader - soft gradient text */
        [data-testid="stMarkdownContainer"] h3 {
            background: linear-gradient(90deg, #9dc0dc 0%, #b8a8d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Success/Error messages - soft colors */
        .stSuccess {
            background-color: rgba(194, 230, 212, 0.3) !important;
            color: #2f6f4f !important;
        }
        
        .stError {
            background-color: rgba(255, 214, 214, 0.3) !important;
            color: #8b3a3a !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔍 X-Ray Baggage Object Detection")

    # Sidebar: model selection
    st.sidebar.header("Model & Ayarlar")
    
    model_type = st.sidebar.selectbox("Model Tipi", ["YOLOv11n", "YOLOv8n", "Faster R-CNN"], index=0)
    
    # Model yükleme
    if model_type == "YOLOv11n":
        weights_path = _find_first_existing(
            PROJECT_ROOT / "runs" / "yolov11n_xray2" / "weights" / "best.pt",
            PROJECT_ROOT / "best.pt",
            PROJECT_ROOT / "best8.pt",
            pattern="best*.pt",
        )

        if not weights_path:
            st.sidebar.error("YOLOv11n için ağırlık dosyası bulunamadı.")
            st.sidebar.info("Projeye `best.pt` yerleştirin veya `runs/.../weights/best.pt` yolunu kullanın.")
            st.stop()

        try:
            model = load_yolo_model(weights_path)
            st.sidebar.success(f"✅ YOLOv11n yüklendi ({Path(weights_path).name})")
            model_info = {"type": "yolo", "model": model}
        except Exception as e:
            st.sidebar.error(f"Model yüklenemedi: {e}")
            st.stop()
    
    elif model_type == "YOLOv8n":
        weights_path = _find_first_existing(
            PROJECT_ROOT / "runs" / "yolov8n_xray4" / "weights" / "best.pt",
            PROJECT_ROOT / "best.pt",
            PROJECT_ROOT / "best8.pt",
            pattern="best*.pt",
        )

        if not weights_path:
            st.sidebar.error("YOLOv8n için ağırlık dosyası bulunamadı.")
            st.sidebar.info("Projeye `best.pt` yerleştirin veya `runs/.../weights/best.pt` yolunu kullanın.")
            st.stop()

        try:
            model = load_yolo_model(weights_path)
            st.sidebar.success(f"✅ YOLOv8n yüklendi ({Path(weights_path).name})")
            model_info = {"type": "yolo", "model": model}
        except Exception as e:
            st.sidebar.error(f"Model yüklenemedi: {e}")
            st.stop()
            
    else:  # Faster R-CNN
        checkpoint_path = _find_first_existing(
            CHECKPOINTS_DIR / "model_epoch_100.pth",
            CHECKPOINTS_DIR,
            pattern="*.pth",
        )

        if not checkpoint_path:
            st.sidebar.error("Faster R-CNN checkpoint dosyası bulunamadı.")
            st.sidebar.info("Lütfen `checkpoints/` içine .pth dosyası koyun veya `main_rcnn.py` ile eğitin.")
            st.stop()
        
        try:
            model, classes, device = load_rcnn_model(checkpoint_path)
            st.sidebar.success(f"✅ Faster R-CNN yüklendi (Epoch 100)")
            model_info = {"type": "rcnn", "model": model, "classes": classes, "device": device}
        except Exception as e:
            st.sidebar.error(f"Model yüklenemedi: {e}")
            st.stop()

    # Parametreler
    conf = st.sidebar.slider("Confidence", 0.0, 1.0, 0.30, 0.01)
    
    if model_type in ["YOLOv11n", "YOLOv8n"]:
        iou = st.sidebar.slider("IoU", 0.0, 1.0, 0.50, 0.01)
        imgsz = st.sidebar.select_slider("Image Size", options=[512, 640, 768, 960], value=640)
    else:
        iou = None
        imgsz = None

    tab_img, tab_vid = st.tabs(["Görüntü", "Video"])

    with tab_img:
        st.subheader("Görüntü Testi")
        
        uploaded = st.file_uploader("Görüntü yükleyin", type=["jpg", "jpeg", "png"])
        use_sample = st.checkbox("Test setinden örnek kullan")
        sample_path = None
        if use_sample:
            samples = sorted(TEST_IMG_DIR.glob("*"))
            if samples:
                sample_path = st.selectbox("Örnek görüntü", samples, index=0, key="sample_select")
            else:
                st.info("Test görüntüsü bulunamadı.")

        # Session state ile son seçimi takip et
        if "last_sample" not in st.session_state:
            st.session_state.last_sample = None
        
        # Örnek değiştiyse veya yeni upload varsa otomatik çalışsın
        auto_run = False
        if use_sample and sample_path and sample_path != st.session_state.last_sample:
            auto_run = True
            st.session_state.last_sample = sample_path
        
        run_btn = st.button("Tahmin Yap (Görüntü)", key="predict_img")
        
        if run_btn or auto_run:
            img_path = None
            original_img = None
            
            if uploaded is not None:
                original_img = Image.open(uploaded)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
                tmp.write(uploaded.getvalue())
                tmp.flush()
                img_path = Path(tmp.name)

            elif sample_path is not None:
                img_path = Path(sample_path)
                original_img = Image.open(img_path)

            else:
                st.warning("Lütfen görüntü yükleyin veya örnek seçin.")
                img_path = None
                original_img = None

            if img_path is not None and original_img is not None:
                with st.spinner("Çalışıyor..."):
                    if model_info["type"] == "yolo":
                        out_img, latency_ms = infer_image_yolo(
                            model_info["model"], img_path, conf, iou, imgsz
                        )
                    else:
                        out_img, latency_ms = infer_image_rcnn(
                            model_info["model"],
                            model_info["classes"],
                            model_info["device"],
                            img_path,
                            conf
                        )

                st.info(f"⏱️ End-to-End Inference Time: {latency_ms:.2f} ms")

                # Yan yana gösterim: orijinal ve sonuç
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_img, caption="Orijinal", use_container_width=True)
                with col2:
                    st.image(out_img, caption="Tahmin Sonucu", use_container_width=True)

    with tab_vid:
        st.subheader("Video Testi")
        
        if model_type not in ["YOLOv11n", "YOLOv8n"]:
            st.info("Video testi şu anda sadece YOLO modelleri için desteklenmektedir.")
        else:
            uploaded_vid = st.file_uploader("Video yükleyin", type=["mp4", "avi", "mov", "mkv"])
            run_vid_btn = st.button("Tahmin Yap (Video)")
            if run_vid_btn:
                if uploaded_vid is None:
                    st.warning("Lütfen video dosyası yükleyin.")
                else:
                    try:
                        # Video dosyasını geçici konuma kaydet
                        tmpv = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_vid.name).suffix)
                        tmpv.write(uploaded_vid.getvalue())
                        tmpv.close()
                        
                        st.info(f"Video yüklendi: {uploaded_vid.name} ({uploaded_vid.size / 1024:.1f} KB)")
                        
                        with st.spinner("Video işleniyor... (500 kare analiz ediliyor, 5 örnek çıkarılıyor)"):
                            sample_frames, frame_count, total_detections = infer_video(
                                model_info["model"], Path(tmpv.name), conf, iou, imgsz
                            )
                        
                        st.success(f"✅ {frame_count} kare işlendi, toplam {total_detections} tespit yapıldı")
                        
                        if sample_frames:
                            st.subheader(f"Örnek Tespit Kareleri ({len(sample_frames)} adet)")
                            for i, frame in enumerate(sample_frames, 1):
                                segment_start = (i - 1) * 100
                                segment_end = i * 100
                                st.image(frame, caption=f"Örnek {i} (Kare {segment_start}-{segment_end})", use_container_width=True)
                        else:
                            st.warning("Videoda tespit bulunamadı. Confidence değerini düşürmeyi deneyin.")
                    except Exception as e:
                        st.error(f"Video işleme hatası: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
