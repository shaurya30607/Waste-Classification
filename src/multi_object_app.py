"""
Standalone multi-object waste classification frontend.

Run from the project root:
    streamlit run src/multi_object_app.py

This is a separate Streamlit app. It does not modify or extend src/app.py.
It uses YOLOv8n for general-object proposals and the existing ResNet-18
checkpoint to classify each detected crop into the project's waste classes.

Changes in this version:
- YOLO now receives a PIL image. Ultralytics treats NumPy arrays as BGR, so
  the previous RGB array reached the detector with red and blue swapped.
- Class-agnostic NMS, so one item cannot get two overlapping boxes with
  different generic labels (for example "bottle" and "vase").
- Stricter defaults (YOLO confidence 0.25, at most 30 objects). The retry
  pass only runs with a confidence lower than the first pass.
- Optional filters for tiny boxes and for generic labels that are never a
  waste item (people, animals, vehicles, street furniture, furniture).
- Phone photos are turned upright using their EXIF orientation.
- Every box is numbered to match its row in the results table; label text
  and line width scale with the image; Windows fonts are found.
- Crops are classified in one batch instead of one at a time.
- Results belong to the uploaded file and the settings that produced them;
  the page says when the settings have changed since the last run.
- The project's own yolov8n.pt is used wherever the app is launched from.
"""

from __future__ import annotations

import functools
import inspect
import io
import math
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import models, transforms
from ultralytics import YOLO


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_expanded_resnet18.pth"
# Ultralytics only downloads the weights if this project copy is missing.
LOCAL_YOLO_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Generic YOLO (COCO) labels that are never a waste item in this project.
# Boxes with these labels are skipped when the matching option is ticked.
NON_WASTE_LABELS = frozenset({
    "person",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe",
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "chair", "couch", "bed", "dining table", "toilet",
})

CONFIDENT_COLOUR = (0, 150, 0)
REVIEW_COLOUR = (220, 120, 0)

FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


# ---------------------------------------------------------------------
# Streamlit compatibility
# ---------------------------------------------------------------------
def full_width(element) -> dict:
    """Keyword arguments that stretch a Streamlit element to its column.

    Newer Streamlit releases use width="stretch"; older ones use
    use_container_width=True. Checking the signature works on both.
    """
    try:
        params = inspect.signature(element).parameters
    except (TypeError, ValueError):
        return {}
    width = params.get("width")
    if width is not None and isinstance(width.default, str):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ResNet-18 classifier...")
def load_classifier(model_path_string: str):
    model_path = Path(model_path_string)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            "Put best_expanded_resnet18.pth in the models folder of the project."
        )

    try:
        checkpoint = torch.load(
            model_path, map_location=DEVICE, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(model_path, map_location=DEVICE)

    classes = list(checkpoint["classes"])
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, len(classes)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    return model, classes


def detector_weights() -> str:
    return str(LOCAL_YOLO_WEIGHTS) if LOCAL_YOLO_WEIGHTS.exists() else "yolov8n.pt"


@st.cache_resource(show_spinner="Loading YOLO object detector...")
def load_detector(weights: str):
    return YOLO(weights)


# ---------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------
def open_upload(uploaded_file) -> Image.Image:
    """Open an upload as an upright RGB image."""
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    try:
        # Phones often store photos sideways plus a "rotate me" EXIF tag.
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass  # Unreadable orientation data: keep the image as stored.
    return image.convert("RGB")


# ---------------------------------------------------------------------
# Detection and classification
# ---------------------------------------------------------------------
def run_detector(detector, image: Image.Image, confidence: float,
                 image_size: int, max_detections: int):
    """Run YOLO once and return its boxes as plain dictionaries."""
    result = detector.predict(
        source=image,          # PIL image: Ultralytics converts RGB to BGR itself
        conf=confidence,
        imgsz=image_size,
        max_det=max_detections,
        agnostic_nms=True,     # one box per object, whatever its generic label
        verbose=False,
    )[0]

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            detections.append({
                "box": tuple(float(v) for v in box.xyxy[0].tolist()),
                "label": str(result.names[int(box.cls[0].item())]),
                "confidence": float(box.conf[0].item()),
            })
    return detections


def clamp_box(box, width: int, height: int):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(math.floor(x1)), width - 1))
    y1 = max(0, min(int(math.floor(y1)), height - 1))
    x2 = max(x1 + 1, min(int(math.ceil(x2)), width))
    y2 = max(y1 + 1, min(int(math.ceil(y2)), height))
    return x1, y1, x2, y2


def filter_detections(detections, width: int, height: int,
                      min_area_percent: float, skip_labels):
    """Drop tiny boxes and boxes whose generic label is never waste."""
    kept = []
    skipped = {"too small": 0, "not a waste item": 0}
    image_area = width * height
    for detection in detections:
        box = clamp_box(detection["box"], width, height)
        if detection["label"] in skip_labels:
            skipped["not a waste item"] += 1
            continue
        area_percent = 100.0 * (box[2] - box[0]) * (box[3] - box[1]) / image_area
        if area_percent < min_area_percent:
            skipped["too small"] += 1
            continue
        kept.append({**detection, "box": box})
    return kept, skipped


def classify_crops(model, classes, crops):
    """Classify all crops in one batch; returns (class, confidence) pairs."""
    batch = torch.stack(
        [CLASSIFIER_TRANSFORM(crop.convert("RGB")) for crop in crops]
    ).to(DEVICE)
    with torch.inference_mode():
        probabilities = F.softmax(model(batch), dim=1).cpu()
    confidences, indices = probabilities.max(dim=1)
    return [
        (classes[index], float(confidence))
        for index, confidence in zip(indices.tolist(), confidences.tolist())
    ]


def analyse_image(image: Image.Image, detect, classify, settings: dict):
    """Detect objects, then classify each one.

    detect(image, confidence) returns detections like run_detector().
    classify(crops) returns one (waste_class, confidence) pair per crop.
    """
    width, height = image.size
    skip_labels = NON_WASTE_LABELS if settings["skip_non_waste"] else frozenset()

    def detect_and_filter(confidence):
        return filter_detections(
            detect(image, confidence), width, height,
            settings["min_area_percent"], skip_labels,
        )

    kept, skipped = detect_and_filter(settings["detector_confidence"])
    retried = False
    if not kept and settings["fallback_confidence"] < settings["detector_confidence"]:
        retried = True
        kept, skipped = detect_and_filter(settings["fallback_confidence"])

    # Crop from the original image, never from the annotated one.
    crops = [image.crop(detection["box"]) for detection in kept]
    predictions = classify(crops) if crops else []

    threshold = settings["classifier_threshold"] / 100.0
    objects = []
    for number, (detection, (waste_class, class_confidence)) in enumerate(
        zip(kept, predictions), start=1
    ):
        objects.append({
            "number": number,
            "box": detection["box"],
            "yolo_label": detection["label"],
            "yolo_confidence": detection["confidence"],
            "waste_class": waste_class,
            "class_confidence": class_confidence,
            "needs_review": class_confidence < threshold,
        })
    return objects, skipped, retried


def table_rows(objects):
    return [
        {
            "Object #": obj["number"],
            "YOLO label": obj["yolo_label"],
            "YOLO confidence (%)": round(obj["yolo_confidence"] * 100, 2),
            "Waste classification": obj["waste_class"],
            "Classifier confidence (%)": round(obj["class_confidence"] * 100, 2),
            "Review required": "Yes" if obj["needs_review"] else "No",
            "Bounding box (x1, y1, x2, y2)": ", ".join(str(v) for v in obj["box"]),
        }
        for obj in objects
    ]


# ---------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------
@functools.lru_cache(maxsize=16)
def load_font(size: int):
    for font_path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow 10.1 and later
    except TypeError:
        return ImageFont.load_default()


def label_text(obj) -> str:
    prefix = f"#{obj['number']} REVIEW: " if obj["needs_review"] else f"#{obj['number']} "
    return f"{prefix}{obj['waste_class']} {obj['class_confidence']:.0%}"


def draw_annotations(image: Image.Image, objects) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    # Keep boxes and text readable on large phone photos.
    scale = max(1.0, max(width, height) / 1000)
    line_width = max(2, round(3 * scale))
    padding = max(2, round(3 * scale))
    font = load_font(min(72, max(14, round(18 * scale))))

    def colour(obj):
        return REVIEW_COLOUR if obj["needs_review"] else CONFIDENT_COLOUR

    # Boxes first, then labels, so a box is never drawn over another label.
    for obj in objects:
        draw.rectangle(obj["box"], outline=colour(obj), width=line_width)

    for obj in objects:
        x1, y1, _, _ = obj["box"]
        text = label_text(obj)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        label_w = right - left + 2 * padding
        label_h = bottom - top + 2 * padding
        label_x = max(0, min(x1, width - label_w))
        # Above the box when there is room, otherwise just inside its top edge.
        label_y = y1 - label_h if y1 - label_h >= 0 else y1
        draw.rectangle(
            (label_x, label_y, label_x + label_w, label_y + label_h),
            fill=colour(obj),
        )
        draw.text(
            (label_x + padding - left, label_y + padding - top),
            text, fill="white", font=font,
        )
    return annotated


# ---------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Multi-Object Waste Classification",
        page_icon="♻️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("♻️ Multi-Object Waste Classification")
    st.caption(
        "Detect multiple objects in one image, classify each detected crop, "
        "and flag uncertain predictions for manual review."
    )
    st.info(
        "Detection uses a pretrained general-purpose YOLO model, not a "
        "waste-specific detector. It can miss waste items or put boxes around "
        "unrelated objects. The ResNet-18 classifier then predicts a waste "
        "category for each box."
    )

    with st.sidebar:
        st.header("⚙️ Detection settings")
        detector_confidence = st.slider(
            "YOLO detection confidence",
            min_value=0.05, max_value=0.75, value=0.25, step=0.05,
            help="Lower values find more objects but add more wrong boxes.",
        )
        fallback_confidence = st.slider(
            "Retry confidence if no objects are kept",
            min_value=0.01, max_value=0.20, value=0.10, step=0.01,
            help=(
                "Used for a second pass only when the first pass keeps no "
                "objects, and only if it is lower than the detection confidence."
            ),
        )
        classifier_threshold = st.slider(
            "Classifier review threshold (%)",
            min_value=40, max_value=95, value=65, step=5,
            help="Waste classifications below this confidence are flagged for review.",
        )
        image_size = st.select_slider(
            "Detection image size",
            options=[640, 800, 960, 1280],
            value=1280,
            help="Larger sizes find smaller objects but run slower.",
        )
        max_detections = st.slider(
            "Maximum objects per image",
            min_value=10, max_value=100, value=30, step=10,
        )

        st.subheader("Filters")
        min_area_percent = st.select_slider(
            "Ignore boxes smaller than",
            options=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            value=0.25,
            format_func=lambda value: f"{value:g}% of the image",
            help="Very small boxes are too blurry for the classifier to judge.",
        )
        skip_non_waste = st.checkbox(
            "Ignore people, animals, vehicles and furniture",
            value=True,
            help="Skips boxes with these YOLO labels: "
                 + ", ".join(sorted(NON_WASTE_LABELS)),
        )

        st.markdown("---")
        st.write(f"**Inference device:** `{DEVICE}`")
        st.write(f"**Classifier checkpoint:** `{MODEL_PATH.name}`")
        st.write(
            "**YOLO weights:** `yolov8n.pt` "
            + ("(project copy)" if LOCAL_YOLO_WEIGHTS.exists() else "(downloads on first run)")
        )

    settings = {
        "detector_confidence": detector_confidence,
        "fallback_confidence": fallback_confidence,
        "classifier_threshold": classifier_threshold,
        "image_size": image_size,
        "max_detections": max_detections,
        "min_area_percent": min_area_percent,
        "skip_non_waste": skip_non_waste,
    }

    left, right = st.columns([1, 1])
    with left:
        uploaded_file = st.file_uploader(
            "Upload an image containing multiple items",
            type=UPLOAD_TYPES,
            help="A clear photo with the items spread apart works best.",
        )
    with right:
        st.markdown("#### What you will get")
        st.markdown(
            "- The image with a numbered box around each detected object\n"
            "- A waste class and classifier confidence for each box\n"
            "- Review flags for low-confidence classifications\n"
            "- Downloads of the annotated image and a CSV report"
        )

    if uploaded_file is None:
        st.markdown("---")
        st.subheader("How it works")
        step1, step2, step3 = st.columns(3)
        step1.markdown("**1. Detect**\n\nYOLO proposes object bounding boxes.")
        step2.markdown("**2. Classify**\n\nResNet-18 classifies each crop.")
        step3.markdown("**3. Review**\n\nLow-confidence predictions are flagged.")
        return

    try:
        original_image = open_upload(uploaded_file)
    except Exception as exc:
        st.error(
            f"This file could not be opened as an image ({exc}). "
            "Try saving it as JPG or PNG and uploading it again."
        )
        return

    file_key = (
        getattr(uploaded_file, "file_id", None)
        or f"{uploaded_file.name}:{uploaded_file.size}"
    )

    st.image(original_image, caption="Uploaded image", **full_width(st.image))

    run_button = st.button(
        "🔍 Detect and classify objects",
        type="primary",
        **full_width(st.button),
    )

    if run_button:
        try:
            with st.spinner("Loading models and analysing the image..."):
                classifier, classes = load_classifier(str(MODEL_PATH))
                detector = load_detector(detector_weights())
                objects, skipped, retried = analyse_image(
                    original_image,
                    detect=lambda image, confidence: run_detector(
                        detector, image, confidence, image_size, max_detections
                    ),
                    classify=lambda crops: classify_crops(classifier, classes, crops),
                    settings=settings,
                )
                annotated = draw_annotations(original_image, objects)

            st.session_state["multi_object_result"] = {
                "file_key": file_key,
                "settings": settings,
                "objects": objects,
                "skipped": skipped,
                "retried": retried,
                "annotated": annotated,
            }
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.exception(exc)

    result = st.session_state.get("multi_object_result")
    if not result or result["file_key"] != file_key:
        return

    st.markdown("---")
    st.subheader("Analysis results")

    if result["settings"] != settings:
        st.info(
            "These results use your previous settings. Click "
            "**Detect and classify objects** again to apply the new ones."
        )

    rows = table_rows(result["objects"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Objects detected", len(rows))
    m2.metric("Flagged for review", sum(row["Review required"] == "Yes" for row in rows))
    m3.metric("Inference device", DEVICE.type.upper())

    if result["retried"]:
        st.warning(
            "The first pass kept no objects, so a second pass used the lower "
            "retry confidence. Expect more wrong boxes than usual."
        )

    skipped = result["skipped"]
    if sum(skipped.values()):
        st.caption(
            f"Ignored {skipped['too small']} box(es) as too small and "
            f"{skipped['not a waste item']} box(es) labelled as people, animals, "
            "vehicles or furniture. Change this under Filters in the sidebar."
        )

    image_col, details_col = st.columns([1.25, 1])
    with image_col:
        st.markdown("#### Annotated image")
        st.image(result["annotated"], **full_width(st.image))
        image_buffer = io.BytesIO()
        result["annotated"].save(image_buffer, format="PNG")
        st.download_button(
            "Download annotated image",
            data=image_buffer.getvalue(),
            file_name="multi_object_annotated.png",
            mime="image/png",
            **full_width(st.download_button),
        )

    with details_col:
        st.markdown("#### Per-object results")
        if rows:
            table = pd.DataFrame(rows)
            st.dataframe(table, hide_index=True, **full_width(st.dataframe))
            st.download_button(
                "Download results CSV",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name="multi_object_results.csv",
                mime="text/csv",
                **full_width(st.download_button),
            )
        else:
            st.warning(
                "No objects were kept. Lower the YOLO detection confidence, "
                "relax the filters in the sidebar, or use a photo where the "
                "items are spread apart."
            )

    st.caption(
        "YOLO confidence measures how sure the detector is that a box holds an "
        "object. Classifier confidence measures how sure ResNet-18 is about the "
        "waste category. They are different scores. Treat predictions as "
        "assistance, not a guarantee of correct disposal."
    )


if __name__ == "__main__":
    main()
