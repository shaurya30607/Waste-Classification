
from pathlib import Path
import csv
import re

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
INPUT_CSV = PROJECT_DIR / "data" / "content_screening_review.csv"
OUTPUT_CSV = PROJECT_DIR / "data" / "graphics_shortlist.csv"

PATTERNS = {
    "infographic": r"infographic|info-graphic",
    "diagram_or_schematic": r"diagram|schematic|flowchart|flow-sheet",
    "chart_or_graph": r"\bchart\b|\bgraph\b|plot-",
    "process_or_lifecycle": r"lifecycle|life-cycle|process-flow|waste-process",
    "instructional_graphic": r"disposal-methods|tabular-form|composition|types-of",
    "academic_figure": r"(^|[-_])fig\d*([_.-]|$)|figure\d*",
}

def main():
    results = []

    with open(INPUT_CSV, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            relative_path = row["file"]
            filename = Path(relative_path).stem.lower()

            matches = [
                label for label, pattern in PATTERNS.items()
                if re.search(pattern, filename)
            ]

            if matches:
                results.append({
                    "file": relative_path,
                    "matched_clues": ", ".join(matches),
                    "model_suggestion": row["suggested_label"],
                    "model_score": row["confidence"],
                    "action": "REVIEW_BEFORE_MOVING"
                })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file", "matched_clues", "model_suggestion",
                "model_score", "action"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Graphics candidates: {len(results)}")
    print(f"Report saved: {OUTPUT_CSV}")
    print("No images were moved or deleted.")

if __name__ == "__main__":
    main()
