
from pathlib import Path
from PIL import Image
import csv
import re

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
REPORT_FILE = DATA_DIR.parent / "image_quality_review.csv"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"
}

# Filename clues only: these are suggestions, not proof.
REVIEW_WORDS = re.compile(
    r"\b(chart|diagram|infographic|poster|flowchart|"
    r"schematic|illustration|graph|statistics|"
    r"banner|logo|icon|educational|report)\b",
    re.IGNORECASE,
)


def main():
    rows = []
    scanned = 0

    for path in DATA_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        scanned += 1
        reasons = []

        if REVIEW_WORDS.search(path.stem.replace("_", " ").replace("-", " ")):
            reasons.append("Filename suggests diagram/chart/graphic")

        try:
            with Image.open(path) as image:
                width, height = image.size

                if width < 100 or height < 100:
                    reasons.append("Very small image")

                if width >= 1200 and height >= 800:
                    reasons.append("Large image; inspect if text-heavy")

        except Exception:
            reasons.append("Image could not be opened")

        if reasons:
            rows.append({
                "file": str(path.relative_to(DATA_DIR)),
                "width": width if "width" in locals() else "",
                "height": height if "height" in locals() else "",
                "reason": "; ".join(reasons),
                "action": "REVIEW MANUALLY"
            })

    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["file", "width", "height", "reason", "action"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Images scanned: {scanned}")
    print(f"Images flagged for review: {len(rows)}")
    print(f"Report saved to: {REPORT_FILE}")
    print("No images were moved or deleted.")


if __name__ == "__main__":
    main()
