
from pathlib import Path
from PIL import Image, ImageDraw
import math

# Locate the project and e-waste folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "data" / "raw" / "Hazardous" / "Hazardous" / "e-waste"

# Save contact sheets separately from the original dataset
OUTPUT_DIR = PROJECT_DIR / "data" / "review" / "e_waste_contact_sheets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

images = sorted(
    path for path in SOURCE_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in EXTENSIONS
)

if not images:
    raise RuntimeError(f"No images found in: {SOURCE_DIR}")

# Contact sheet settings
BATCH_SIZE = 100
COLUMNS = 5
THUMB_WIDTH = 150
THUMB_HEIGHT = 110
LABEL_HEIGHT = 24
PADDING = 8

CELL_WIDTH = THUMB_WIDTH + PADDING * 2
CELL_HEIGHT = THUMB_HEIGHT + LABEL_HEIGHT + PADDING * 2

print(f"Found {len(images)} images")

for start in range(0, len(images), BATCH_SIZE):
    batch = images[start:start + BATCH_SIZE]
    rows = math.ceil(len(batch) / COLUMNS)

    sheet = Image.new(
        "RGB",
        (COLUMNS * CELL_WIDTH, rows * CELL_HEIGHT),
        "white"
    )
    draw = ImageDraw.Draw(sheet)

    for position, image_path in enumerate(batch):
        index = start + position
        column = position % COLUMNS
        row = position // COLUMNS

        x = column * CELL_WIDTH + PADDING
        y = row * CELL_HEIGHT + PADDING

        try:
            with Image.open(image_path) as original:
                image = original.convert("RGB")
                image.thumbnail((THUMB_WIDTH, THUMB_HEIGHT))

            sheet.paste(image, (x, y))
            draw.text(
                (x, y + THUMB_HEIGHT + 3),
                f"{index}: {image_path.name[:18]}",
                fill="black"
            )
        except Exception as error:
            draw.text((x, y), f"Could not open: {index}", fill="red")
            print(f"Skipped {image_path.name}: {error}")

    sheet_number = start // BATCH_SIZE + 1
    output_path = OUTPUT_DIR / f"e_waste_sheet_{sheet_number:02d}.jpg"
    sheet.save(output_path, quality=90)

    print(f"Created: {output_path}")

print(f"\nDone! Review sheets are in:\n{OUTPUT_DIR}")
