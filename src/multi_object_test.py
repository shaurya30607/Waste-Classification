"""Random-dataset multi-object waste classification test.

From the project root, run:
    python3 src/multi_object_test.py

Requires:
    pip install ultralytics

The script selects one random image under data/raw/, detects objects with a
pretrained YOLO model, and classifies each detected crop using the existing
ResNet-18 checkpoint. The pretrained detector is not waste-specific, so it may
miss some waste items or produce imperfect boxes.
"""
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = PROJECT_ROOT / "models" / "best_expanded_resnet18.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "multi_object_results"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

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


def choose_random_dataset_image() -> Path:
    """Choose a random image recursively from data/raw/."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}\n"
            "Run this script from src/multi_object_test.py within your project, "
            "and make sure data/raw/ exists."
        )

    images = sorted(
        path for path in DATA_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.stem.endswith("_multi_object_results")
    )
    if not images:
        raise FileNotFoundError(
            f"No supported images found under {DATA_DIR}. "
            "Expected JPG, JPEG, PNG, WEBP, or BMP files."
        )

    selected = random.choice(images)
    print(f"Found {len(images)} dataset image(s).")
    print(f"Randomly selected: {selected.relative_to(PROJECT_ROOT)}")
    return selected


def load_classifier():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}\n"
            "Make sure models/best_expanded_resnet18.pth exists."
        )

    # weights_only=False is needed for checkpoints that include class metadata
    # along with tensors (supported by current PyTorch releases).
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions without weights_only.
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    classes = checkpoint["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, len(classes)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    return model, classes


def classify_crop(model, classes, crop):
    tensor = CLASSIFIER_TRANSFORM(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = F.softmax(model(tensor), dim=1)[0]
    index = int(torch.argmax(probabilities).item())
    return classes[index], float(probabilities[index].item())


def main():
    # Lower detector confidence and a larger inference size help with small or
    # crowded objects. A lower threshold can also create extra false positives.
    detector_confidence = 0.10
    fallback_confidence = 0.05
    detection_image_size = 1280
    classifier_threshold = 0.65
    yolo_model_name = "yolov8n.pt"

    image_path = choose_random_dataset_image()

    print(f"\nDevice: {DEVICE}")
    print(f"Loading classifier checkpoint: {MODEL_PATH}")
    classifier, classes = load_classifier()
    print(f"Classifier classes: {classes}")

    print(f"Loading YOLO detector: {yolo_model_name}")
    # On the first run, Ultralytics may download the YOLO model weights.
    detector = YOLO(yolo_model_name)

    with Image.open(image_path) as source_image:
        original_image = source_image.convert("RGB")
    image = original_image.copy()

    result = detector.predict(
        source=np.array(original_image),
        conf=detector_confidence,
        imgsz=detection_image_size,
        max_det=100,
        verbose=False,
    )[0]

    # If the first pass finds no objects, retry more permissively. This may
    # recover faint detections in cluttered images, but can add false positives.
    if result.boxes is None or len(result.boxes) == 0:
        print(
            f"No detections at confidence {detector_confidence:.2f}; "
            f"retrying at {fallback_confidence:.2f}..."
        )
        result = detector.predict(
            source=np.array(original_image),
            conf=fallback_confidence,
            imgsz=detection_image_size,
            max_det=100,
            verbose=False,
        )[0]

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    detections = result.boxes
    count = 0

    if detections is not None and len(detections) > 0:
        for box in detections:
            coords = box.xyxy[0].detach().cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords.tolist()
            x1 = max(0, min(x1, image.width - 1))
            y1 = max(0, min(y1, image.height - 1))
            x2 = max(x1 + 1, min(x2, image.width))
            y2 = max(y1 + 1, min(y2, image.height))

            # Always crop from the untouched source, not the annotated output.
            crop = original_image.crop((x1, y1, x2, y2))
            predicted_class, class_conf = classify_crop(classifier, classes, crop)
            detector_conf = float(box.conf[0].item())
            detector_name = detector.names[int(box.cls[0].item())]
            count += 1

            if class_conf >= classifier_threshold:
                label = f"{predicted_class} {class_conf * 100:.0f}%"
                color = (0, 150, 0)
            else:
                label = f"REVIEW: {predicted_class} {class_conf * 100:.0f}%"
                color = (220, 120, 0)

            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
            # Keep the text label inside the image as much as possible.
            label_top = max(0, top)
            draw.rectangle((left, label_top, right + 4, bottom + 4), fill=color)
            draw.text((x1 + 2, label_top + 2), label, fill="white", font=font)

            print(
                f"Object {count}: detector={detector_name} ({detector_conf:.1%}), "
                f"waste_prediction={predicted_class} ({class_conf:.1%})"
            )

    if count == 0:
        print(
            "No objects detected even after the lower-confidence retry. "
            "The pretrained YOLO model is a general-object detector, not a "
            "waste-specific detector, and dense/overlapping bottles can be missed. "
            "Try another image or a clearer image with separated objects."
        )
    else:
        print(f"\nDetected and classified {count} object(s).")
        print("Detector confidence and classifier confidence measure different things.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{image_path.stem}_multi_object_results.jpg"
    # Avoid overwriting a previous result when filenames repeat.
    suffix = 2
    while output_path.exists():
        output_path = OUTPUT_DIR / f"{image_path.stem}_multi_object_results_{suffix}.jpg"
        suffix += 1
    image.save(output_path, quality=95)
    print(f"Annotated result saved to: {output_path}")

    # On macOS, open the result so it is easy to inspect after running.
    if sys.platform == "darwin":
        subprocess.run(["open", str(output_path)], check=False)


if __name__ == "__main__":
    main()
