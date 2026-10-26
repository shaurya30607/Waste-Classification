
from pathlib import Path
import hashlib
import shutil
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
REVIEW_DIR = DATA_DIR.parent / "duplicates_review"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"
}


def calculate_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def main():
    groups = defaultdict(list)

    for path in DATA_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            groups[calculate_hash(path)].append(path)

    moved = 0

    for paths in groups.values():
        if len(paths) < 2:
            continue

        # Group exact duplicates by their top-level category.
        by_category = defaultdict(list)

        for path in paths:
            relative = path.relative_to(DATA_DIR)
            category = relative.parts[0] if len(relative.parts) > 1 else "Unknown"
            by_category[category].append(path)

        for category_paths in by_category.values():
            # Keep one copy in each category.
            category_paths.sort(key=lambda p: ("copy" in p.stem.lower(), str(p)))
            for duplicate in category_paths[1:]:
                destination = REVIEW_DIR / duplicate.relative_to(DATA_DIR)
                destination.parent.mkdir(parents=True, exist_ok=True)

                # Avoid overwriting an existing file.
                if destination.exists():
                    destination = destination.with_name(
                        f"{destination.stem}_duplicate{destination.suffix}"
                    )

                shutil.move(str(duplicate), str(destination))
                moved += 1

    print(f"Duplicate files moved to review folder: {moved}")
    print(f"Review folder: {REVIEW_DIR}")
    print("One copy per category has been retained.")


if __name__ == "__main__":
    main()
