from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def extract_ivp_features(image_path: str):
    """
    Extracts hand-crafted IVP features:
    - HSV Color Histograms (Color distribution)
    - Sobel Edge Gradient Statistics (Texture & Edges)
    - Grayscale Intensity Statistics (Brightness / Contrast)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    img = cv2.resize(img, (128, 128))

    # 1. Color Features (HSV Space)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
    color_features = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    color_features = color_features / (np.sum(color_features) + 1e-6)  # Normalization

    # 2. Edge & Gradient Features (Sobel Filters)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    edge_mean = np.mean(gradient_magnitude)
    edge_std = np.std(gradient_magnitude)

    # 3. Statistical Intensity Features
    gray_mean = np.mean(gray)
    gray_std = np.std(gray)

    # Combine into a single feature vector
    feature_vector = np.hstack([
        color_features,
        [edge_mean, edge_std, gray_mean, gray_std]
    ])
    return feature_vector


def load_dataset_features():
    classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    X, y = [], []
    print("Extracting IVP features from dataset...")
    for cls_name in classes:
        cls_folder = DATA_DIR / cls_name
        images = list(cls_folder.rglob("*.jpg")) + list(cls_folder.rglob("*.png")) + list(cls_folder.rglob("*.jpeg"))

        for img_path in tqdm(images, desc=f"Processing {cls_name}"):
            feats = extract_ivp_features(img_path)
            if feats is not None:
                X.append(feats)
                y.append(class_to_idx[cls_name])

    return np.array(X), np.array(y), classes


def main():
    X, y, classes = load_dataset_features()

    if len(X) == 0:
        print("No features extracted. Ensure images are in data/raw.")
        return

    # Train / Test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nFeature vector dimension per image: {X.shape[1]}")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 1. Train Decision Tree (Unit 5 Syllabus)
    print("\n--- Training Decision Tree Classifier ---")
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    print(classification_report(y_test, dt_preds, target_names=classes))

    # 2. Train Random Forest Classifier
    print("\n--- Training Random Forest Classifier ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    print(classification_report(y_test, rf_preds, target_names=classes))

    # Plot Confusion Matrix for Random Forest
    cm = confusion_matrix(y_test, rf_preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix (IVP Features + Random Forest)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()