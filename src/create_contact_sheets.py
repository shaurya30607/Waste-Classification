
from pathlib import Path
import csv
import math
from PIL import Image, ImageDraw
import textwrap

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
REPORT_FILE = PROJECT_DIR / "data" / "image_quality_review.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "contact_sheets"

THUMB_WIDTH = 180
THUMB_HEIGHT = 145
CELL_WIDTH = 210
CELL_HEIGHT = 195
COLUMNS = 4
IMAGES_PER_SHEET = 40


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    # Remove old generated sheets so they don't get mixed with new ones.
    for old_file in OUTPUT_DIR.glob("contact_sheet_*.jpg"):
        old_file.unlink()

    paths = []
    for row in rows:
        path = DATA_DIR / row["file"]
        if path.is_file():
            paths.append((path, row["reason"]))

    total_sheets = math.ceil(len(paths) / IMAGES_PER_SHEET)

    for sheet_index in range(total_sheets):
        batch = paths[
            sheet_index * IMAGES_PER_SHEET:
            (sheet_index + 1) * IMAGES_PER_SHEET
        ]

        rows_count = math.ceil(len(batch) / COLUMNS)
        sheet = Image.new(
            "RGB",
            (COLUMNS * CELL_WIDTH, rows_count * CELL_HEIGHT),
            "white"
        )
        draw = ImageDraw.Draw(sheet)

        for index, (path, reason) in enumerate(batch):
            x = (index % COLUMNS) * CELL_WIDTH
            y = (index // COLUMNS) * CELL_HEIGHT

            try:
                with Image.open(path) as original:
                    image = original.convert("RGB")
                    image.thumbnail((THUMB_WIDTH, THUMB_HEIGHT))
                    image_x = x + (CELL_WIDTH - image.width) // 2
                    sheet.paste(image, (image_x, y + 5))

                filename = str(path.relative_to(DATA_DIR))
                lines = textwrap.wrap(filename, width=30)[:3]

                for line_number, line in enumerate(lines):
                    draw.text(
                        (x + 5, y + 153 + line_number * 12),
                        line,
                        fill="black"
                    )

                draw.text(
                    (x + 5, y + 190 - 12),
                    reason[:30],
                    fill="darkred"
                )

            except Exception:
                draw.text((x + 5, y + 10), "Could not open image", fill="red")

        output_path = OUTPUT_DIR / f"contact_sheet_{sheet_index + 1:02d}.jpg"
        sheet.save(output_path, quality=90)
        print(f"Created: {output_path}")

    print(f"\nImages included: {len(paths)}")
    print(f"Contact sheets created: {total_sheets}")
    print("Original dataset unchanged.")


if __name__ == "__main__":
    main()
