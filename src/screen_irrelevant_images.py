
from pathlib import Path
import csv
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
REPORT_FILE = PROJECT_DIR / "data" / "content_screening_review.csv"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CANDIDATES = [
    "a photograph of discarded waste materials",
    "a photograph of broken or used household items being discarded",
    "a photograph of recyclable waste such as paper, glass, metal or plastic",
    "a photograph of organic food scraps or garden waste",
    "an educational infographic with text and recycling symbols",
    "a chart, diagram, or flowchart about waste management",
    "a recycling instruction poster with text and icons",
    "a decorative product catalogue photograph of intact pottery or ceramics",
    "a photograph of a new consumer product on a display background",
    "a logo or illustration rather than a photograph",
]

# CLIP is a pretrained image-text model.
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 16


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    images = [
        path for path in DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    ]

    rows = []

    for start in range(0, len(images), BATCH_SIZE):
        batch_paths = images[start:start + BATCH_SIZE]
        valid_images = []
        valid_paths = []

        for path in batch_paths:
            try:
                with Image.open(path) as image:
                    valid_images.append(image.convert("RGB"))
                valid_paths.append(path)
            except Exception:
                rows.append({
                    "file": str(path.relative_to(DATA_DIR)),
                    "suggested_label": "UNREADABLE_IMAGE",
                    "confidence": "",
                    "action": "REVIEW"
                })

        if not valid_images:
            continue

        inputs = processor(
            text=CANDIDATES,
            images=valid_images,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = outputs.logits_per_image.softmax(dim=1)

        for path, scores in zip(valid_paths, probabilities):
            best_index = int(scores.argmax())
            label = CANDIDATES[best_index]
            confidence = float(scores[best_index])

            rows.append({
                "file": str(path.relative_to(DATA_DIR)),
                "suggested_label": label,
                "confidence": round(confidence, 4),
                "action": "REVIEW" if best_index >= 4 else "LIKELY_KEEP"
            })

        processed = min(start + BATCH_SIZE, len(images))
        print(f"Processed {processed}/{len(images)} images")

    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file", "suggested_label", "confidence", "action"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nReport saved: {REPORT_FILE}")
    print(f"Images processed: {len(rows)}")
    print("No images were moved or deleted.")
    print("Model suggestions are imperfect; review before removing files.")


if __name__ == "__main__":
    main()
