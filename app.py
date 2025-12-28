import os
import tempfile
from pathlib import Path
import time

import numpy as np
import streamlit as st
from PIL import Image
import torch

# RCNN utilities
from main_rcnn import build_model, infer_classes_from_xmls, DATA_ROOT as RCNN_DATA_ROOT


# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"
TEST_IMG_DIR = DATA_ROOT
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


# ============================================================
# MODEL LOADERS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    from ultralytics import YOLO  # lazy import (Cloud-safe)
    return YOLO(weights_path)


@st.cache_resource(show_spinner=False)
def load_rcnn_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    annotations_path = os.path.join(RCNN_DATA_ROOT, "Annotations")
    classes = infer_classes_from_xmls(annotations_path)
    num_classes = len(classes) + 1  # background

    model = build_model(num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes, device


# ============================================================
# YOLO IMAGE INFERENCE
# ============================================================
def infer_image_yolo(model, img_path: Path, conf: float, iou: float, imgsz: int):
    t0 = time.perf_counter()

    results = model.predict(
        source=str(img_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )

    arr = results[0].plot()       # BGR
    arr = arr[:, :, ::-1]         # RGB
    out_img = Image.fromarray(arr)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return out_img, latency_ms


# ============================================================
# FASTER R-CNN IMAGE INFERENCE
# ============================================================
def infer_image_rcnn(model, classes, device, img_path: Path, conf: float):
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import io

    t0 = time.perf_counter()

    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = T.ToTensor()(img_pil).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    mask = scores >= conf
    boxes, labels, scores = boxes[mask], labels[mask], scores[mask]

    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(img_pil)

    colors = ["red", "blue", "green", "yellow", "cyan", "magenta", "orange"]
    class_names = ["__background__"] + classes

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        color = colors[int(label) % len(colors)]

        ax.add_patch(
            patches.Rectangle((x1, y1), w, h, linewidth=3,
                              edgecolor=color, facecolor="none")
        )

        ax.text(
            x1, max(0, y1 - 10),
            f"{class_names[int(label)]}: {score:.2f}",
            bbox=dict(facecolor=color, alpha=0.8),
            fontsize=13, color="white"
        )

    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    out_img = Image.open(buf).copy()
    plt.close(fig)
    buf.close()

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return out_img, latency_ms


# ============================================================
# YOLO VIDEO INFERENCE (SAFE)
# ============================================================
def infer_video(model, video_path: Path, conf: float, iou: float, imgsz: int):
    results = model.predict(
        source=str(video_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        stream=True,
        verbose=False,
    )

    sample_frames = []
    frame_count = 0
    total_detections = 0

    for result in results:
        frame_count += 1
        if result.boxes is None:
            continue

        total_detections += len(result.boxes)

        if len(sample_frames) < 5:
            arr = result.plot()
            arr = arr[:, :, ::-1]
            sample_frames.append(Image.fromarray(arr))

        if frame_count >= 500:
            break

    return sample_frames, frame_count, total_detections


# ============================================================
# STREAMLIT UI
# ============================================================
def main():
    st.set_page_config(
        page_title="X-Ray Baggage Detection",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.title("🔍 X-Ray Baggage Object Detection")

    # Sidebar
    st.sidebar.header("Model & Ayarlar")
    model_type = st.sidebar.selectbox(
        "Model Tipi", ["YOLOv11n", "YOLOv8n", "Faster R-CNN"]
    )

    # Load model
    if model_type == "YOLOv11n":
        weights_path = PROJECT_ROOT / "runs/yolov11n_xray2/weights/best.pt"
        model = load_yolo_model(str(weights_path))
        model_info = {"type": "yolo", "model": model}

    elif model_type == "YOLOv8n":
        weights_path = PROJECT_ROOT / "runs/yolov8n_xray4/weights/best.pt"
        model = load_yolo_model(str(weights_path))
        model_info = {"type": "yolo", "model": model}

    else:
        checkpoint_path = CHECKPOINTS_DIR / "model_epoch_100.pth"
        model, classes, device = load_rcnn_model(str(checkpoint_path))
        model_info = {
            "type": "rcnn",
            "model": model,
            "classes": classes,
            "device": device
        }

    # Params
    conf = st.sidebar.slider("Confidence", 0.0, 1.0, 0.3, 0.01)

    if model_type != "Faster R-CNN":
        iou = st.sidebar.slider("IoU", 0.0, 1.0, 0.5, 0.01)
        imgsz = st.sidebar.select_slider("Image Size", [512, 640, 768], value=640)
    else:
        iou, imgsz = None, None

    tab_img, tab_vid = st.tabs(["Görüntü", "Video"])

    # ---------------- IMAGE ----------------
    with tab_img:
        uploaded = st.file_uploader("Görüntü yükleyin", ["jpg", "jpeg", "png"])

        if st.button("Tahmin Yap"):
            if uploaded:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                tmp.write(uploaded.getvalue())
                tmp.close()

                original = Image.open(tmp.name)

                with st.spinner("Inference..."):
                    if model_info["type"] == "yolo":
                        out, t = infer_image_yolo(
                            model_info["model"], Path(tmp.name), conf, iou, imgsz
                        )
                    else:
                        out, t = infer_image_rcnn(
                            model_info["model"],
                            model_info["classes"],
                            model_info["device"],
                            Path(tmp.name),
                            conf
                        )

                st.info(f"⏱ {t:.2f} ms")
                c1, c2 = st.columns(2)
                c1.image(original, caption="Orijinal", use_container_width=True)
                c2.image(out, caption="Tahmin", use_container_width=True)

    # ---------------- VIDEO ----------------
    with tab_vid:
        if model_type == "Faster R-CNN":
            st.info("Video sadece YOLO için destekleniyor.")
        else:
            uploaded_vid = st.file_uploader("Video yükleyin", ["mp4", "avi", "mov"])

            if st.button("Video Tahmin"):
                if uploaded_vid:
                    tmpv = tempfile.NamedTemporaryFile(delete=False)
                    tmpv.write(uploaded_vid.getvalue())
                    tmpv.close()

                    with st.spinner("Video işleniyor..."):
                        frames, fc, dets = infer_video(
                            model_info["model"], Path(tmpv.name), conf, iou, imgsz
                        )

                    st.success(f"{fc} kare, {dets} tespit")
                    for i, f in enumerate(frames):
                        st.image(f, caption=f"Örnek {i+1}", use_container_width=True)


if __name__ == "__main__":
    main()