import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TEMP_DIR = PROJECT_ROOT / "data" / "temp_trashnet"

TRASHNET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"

def setup_trashnet():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TEMP_DIR / "dataset-resized.zip"

    print("Step 1: Downloading TrashNet dataset archive (~40MB)...")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(TRASHNET_URL, zip_path)
    print("Download complete.")

    print("\nStep 2: Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(TEMP_DIR)

    extracted_base = TEMP_DIR / "dataset-resized"
    if not extracted_base.exists():
        # Handle cases where archive extracts without top-level wrapper
        extracted_base = TEMP_DIR

    # Mapping TrashNet 6 classes to your 4 project classes
    class_mapping = {
        "cardboard": RAW_DATA_DIR / "Recyclable" / "cardboard",
        "glass":     RAW_DATA_DIR / "Recyclable" / "glass",
        "metal":     RAW_DATA_DIR / "Recyclable" / "metal",
        "paper":     RAW_DATA_DIR / "Recyclable" / "paper",
        "plastic":   RAW_DATA_DIR / "Recyclable" / "plastic",
        "trash":     RAW_DATA_DIR / "Non-Recyclable" / "trashnet_trash"
    }

    print("\nStep 3: Integrating images into project categories...")
    total_added = 0
    for src_folder_name, dest_path in class_mapping.items():
        src_path = extracted_base / src_folder_name
        if not src_path.exists():
            continue

        dest_path.mkdir(parents=True, exist_ok=True)
        images = list(src_path.glob("*.jpg")) + list(src_path.glob("*.png"))

        for img_file in images:
            target_file = dest_path / f"tn_{img_file.name}"
            shutil.copy(img_file, target_file)
            total_added += 1

    print(f"Successfully added {total_added} annotated images into data/raw/.")

    # Cleanup temporary archive
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("Temporary download files cleaned up.")

if __name__ == "__main__":
    setup_trashnet()