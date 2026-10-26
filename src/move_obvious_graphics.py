
from pathlib import Path
import csv
import shutil

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
INPUT_CSV = PROJECT_DIR / "data" / "graphics_shortlist.csv"
REVIEW_DIR = PROJECT_DIR / "data" / "graphics_review"

STRONG_CLUES = (
    "infographic",
    "info-graphic",
    "schematic",
    "flow-sheet",
    "flowchart",
    "lifecycle",
    "life-cycle",
    "diagram-showing",
    "waste-process",
)

def main():
    with open(INPUT_CSV, newline="", encoding="utf-8") as file:
        candidates = list(csv.DictReader(file))

    selected = [
        row for row in candidates
        if any(clue in Path(row["file"]).name.lower()
               for clue in STRONG_CLUES)
    ]

    print(f"Strong candidates: {len(selected)}")
    print("Moving candidates to a separate review folder...")

    moved = 0

    for row in selected:
        source = DATA_DIR / row["file"]

        if not source.is_file():
            print(f"Skipped missing file: {row['file']}")
            continue

        destination = REVIEW_DIR / row["file"]
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            print(f"Skipped existing destination: {row['file']}")
            continue

        shutil.move(str(source), str(destination))
        moved += 1

    print(f"Files moved: {moved}")
    print(f"Review folder: {REVIEW_DIR}")
    print("Files were moved, not permanently deleted.")

if __name__ == "__main__":
    main()
