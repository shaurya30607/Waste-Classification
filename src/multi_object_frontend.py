"""Streamlit UI for multi-object waste detection and classification.

Import and call render_multi_object_section() from src/app.py. It reuses the
ResNet-18 model already loaded by the dashboard and caches the YOLO detector.
"""
from __future__ import annotations

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ultralytics import YOLO


IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp"]
CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@st.cache_resource
def load_detector(model_name: str = "yolov8n.pt"):
    """Load YOLO once per Streamlit session process."""
    return YOLO(model_name)


def _classify_crop(model, classes, crop: Image.Image, device):
    tensor = CLASSIFIER_TRANSFORM(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = F.softmax(model(tensor), dim=1)[0]
    index = int(torch.argmax(probabilities).item())
    return str(classes[index]), float(probabilities[index].item())


def _run_detection(
    image: Image.Image,
    classifier,
    classes,
    device,
    detector_confidence: float,
    fallback_confidence: float,
    image_size: int,
    classifier_threshold: float,
):
    """Detect objects and classify each detected crop; return image and rows."""
    original = image.convert("RGB")
    annotated = original.copy()
    detector = load_detector()

    result = detector.predict(
        source=np.array(original),
        conf=detector_confidence,
        imgsz=image_size,
        max_det=100,
        verbose=False,
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        result = detector.predict(
            source=np.array(original),
            conf=fallback_confidence,
            imgsz=image_size,
            max_det=100,
            verbose=False,
        )[0]

    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    rows = []
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            coords = box.xyxy[0].detach().cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords.tolist()
            x1 = max(0, min(x1, original.width - 1))
            y1 = max(0, min(y1, original.height - 1))
            x2 = max(x1 + 1, min(x2, original.width))
            y2 = max(y1 + 1, min(y2, original.height))

            crop = original.crop((x1, y1, x2, y2))
            predicted_class, class_confidence = _classify_crop(
                classifier, classes, crop, device
            )
            detector_confidence_value = float(box.conf[0].item())
            detector_label = str(result.names[int(box.cls[0].item())])
            review = class_confidence < classifier_threshold
            label = (
                f"REVIEW: {predicted_class} {class_confidence:.0%}"
                if review else f"{predicted_class} {class_confidence:.0%}"
            )
            color = (220, 120, 0) if review else (0, 150, 0)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
            label_top = max(0, top)
            draw.rectangle((left, label_top, right + 4, bottom + 4), fill=color)
            draw.text((x1 + 2, label_top + 2), label, fill="white", font=font)

            rows.append({
                "Object #": len(rows) + 1,
                "Waste classification": predicted_class,
                "Classifier confidence": round(class_confidence * 100, 2),
                "YOLO object label": detector_label,
                "YOLO confidence": round(detector_confidence_value * 100, 2),
                "Status": "Review recommended" if review else "Above threshold",
                "Bounding box (x1, y1, x2, y2)": f"({x1}, {y1}, {x2}, {y2})",
            })

    return annotated, rows


def render_multi_object_section(model, classes, device, default_threshold: int = 65):
    """Render the separate multi-object feature in the Streamlit dashboard."""
    st.markdown("---")
    st.header("🔎 Multi-Object Waste Detection")
    st.write(
        "Upload a photo containing multiple items. YOLO proposes object regions, "
        "then the existing ResNet-18 model classifies each crop."
    )
    st.info(
        "Limitations: YOLOv8n is a general-purpose detector, not a waste-specific "
        "detector. It can miss items or create false detections. The YOLO label "
        "is the detector's generic object category; the waste category comes from ResNet-18."
    )

    if model is None or not classes:
        st.error("The ResNet-18 checkpoint is not loaded. Multi-object analysis is unavailable.")
        return

    left, right = st.columns([2, 1])
    with left:
        uploaded = st.file_uploader(
            "Upload an image with multiple objects",
            type=IMAGE_EXTENSIONS,
            key="multi_object_upload",
        )
    with right:
        threshold = st.slider(
            "Flag predictions below (%)",
            min_value=40,
            max_value=95,
            value=int(default_threshold),
            step=5,
            key="multi_object_threshold",
            help="Low-confidence crop classifications are marked for manual review.",
        )

    detector_col, size_col = st.columns(2)
    with detector_col:
        detector_conf = st.slider(
            "YOLO detection confidence (%)", 5, 50, 10, 5,
            key="multi_object_detector_conf",
            help="Lower values may find more objects but can add false detections.",
        )
    with size_col:
        image_size = st.select_slider(
            "Detection image size",
            options=[640, 960, 1280],
            value=1280,
            key="multi_object_image_size",
        )

    if uploaded is None:
        st.caption("Choose an image to enable multi-object analysis.")
        return

    try:
        source_image = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"Could not open the uploaded image: {exc}")
        return

    st.image(source_image, caption="Original image", use_container_width=True)
    if st.button("Detect and classify objects", type="primary", key="run_multi_object"):
        with st.spinner("Loading detector and analysing object crops…"):
            try:
                annotated, rows = _run_detection(
                    image=source_image,
                    classifier=model,
                    classes=classes,
                    device=device,
                    detector_confidence=detector_conf / 100,
                    fallback_confidence=0.05,
                    image_size=int(image_size),
                    classifier_threshold=threshold / 100,
                )
            except Exception as exc:
                st.error(f"Multi-object analysis failed: {exc}")
                st.caption("Check that ultralytics is installed and that the YOLO weights can be loaded.")
                return

        st.session_state["multi_object_result_image"] = annotated
        st.session_state["multi_object_result_rows"] = rows
        st.session_state["multi_object_result_name"] = uploaded.name

    rows = st.session_state.get("multi_object_result_rows")
    result_image = st.session_state.get("multi_object_result_image")
    result_name = st.session_state.get("multi_object_result_name", "uploaded image")
    if rows is not None and result_image is not None:
        st.subheader("Detection results")
        st.image(result_image, caption=f"Annotated result — {result_name}", use_container_width=True)
        total = len(rows)
        needs_review = sum(row["Status"] == "Review recommended" for row in rows)
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Objects classified", total)
        metric2.metric("Review recommended", needs_review)
        metric3.metric("Above threshold", total - needs_review)

        if total:
            st.dataframe(rows, use_container_width=True, hide_index=True)
            import pandas as pd
            csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download detection results (CSV)",
                data=csv_bytes,
                file_name="multi_object_detection_results.csv",
                mime="text/csv",
                key="download_multi_object_csv",
            )
        else:
            st.warning(
                "No objects were detected. Try a clearer image with separated items, "
                "or lower the YOLO confidence threshold."
            )
