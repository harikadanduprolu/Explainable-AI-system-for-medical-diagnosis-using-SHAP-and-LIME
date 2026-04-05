#!/usr/bin/env python3
"""Train a local CXR model on downloaded MIMIC-CXR p10 data and generate Grad-CAM visualizations.

This script expects a locally downloaded `p10` folder plus the CheXpert and split CSVs.
It trains the same lightweight TensorFlow CNN used by `train_mimic_cxr_from_gcs.py` and
then runs Grad-CAM on a few matched local images.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as keras_image
except Exception as exc:  # pragma: no cover - handled at runtime
    tf = None
    keras_image = None
    TF_IMPORT_ERROR = exc
else:
    TF_IMPORT_ERROR = None

TF_AVAILABLE = tf is not None


def load_csv_from_path(path: str) -> pd.DataFrame:
    if path.endswith(".gz"):
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def build_dicom_label_map_local(
    chexpert_local: str,
    split_local: str,
    label_column: str,
    uncertain_policy: str,
) -> Dict[str, int]:
    chexpert_df = load_csv_from_path(chexpert_local)

    if label_column not in chexpert_df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in {chexpert_local}. "
            f"Available columns: {list(chexpert_df.columns)}"
        )

    if "dicom_id" in chexpert_df.columns:
        merged = chexpert_df.copy()
    else:
        split_df = load_csv_from_path(split_local)
        required_cols = {"subject_id", "study_id", "dicom_id"}
        if not required_cols.issubset(set(split_df.columns)):
            raise ValueError(
                f"Split CSV missing required columns {required_cols}. Found: {set(split_df.columns)}"
            )
        if not {"subject_id", "study_id"}.issubset(set(chexpert_df.columns)):
            raise ValueError(
                "CheXpert CSV missing subject_id/study_id required for merge with split CSV."
            )
        merged = split_df[["subject_id", "study_id", "dicom_id"]].merge(
            chexpert_df,
            on=["subject_id", "study_id"],
            how="inner",
        )

    label_series = merged[label_column]
    if uncertain_policy == "ones":
        label_series = label_series.replace(-1, 1)
    elif uncertain_policy == "zeros":
        label_series = label_series.replace(-1, 0)
    else:
        label_series = label_series.replace(-1, np.nan)

    valid = merged.assign(_label=label_series).dropna(subset=["dicom_id", "_label"])
    valid = valid[valid["_label"].isin([0, 1])]

    dicom_to_label = {
        str(dicom_id): int(label)
        for dicom_id, label in zip(valid["dicom_id"], valid["_label"])
    }
    if not dicom_to_label:
        raise RuntimeError("No valid labels after processing. Check the label column and uncertain policy.")
    return dicom_to_label


def build_model(img_size: int):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is unavailable in this environment")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(img_size, img_size, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def load_batch_local(
    local_root: str,
    img_size: int,
    dicom_label_map: Dict[str, int],
    limit: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    root = Path(local_root)
    candidates = list(root.rglob("*.jpg")) + list(root.rglob("*.jpeg")) + list(root.rglob("*.png"))

    images: List[np.ndarray] = []
    labels: List[int] = []

    for path in candidates:
        dicom_id = path.stem
        if dicom_id not in dicom_label_map:
            continue

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)

        images.append(img)
        labels.append(dicom_label_map[dicom_id])

        if limit > 0 and len(images) >= limit:
            break

    if not images:
        return (
            np.empty((0, img_size, img_size, 1), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


class GradCAM:
    def __init__(self, model, layer_name: Optional[str] = None):
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        self.grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output],
        )

    def _find_last_conv_layer(self) -> str:
        for layer in reversed(self.model.layers):
            if "conv" in layer.name.lower():
                return layer.name
        raise ValueError("No convolutional layer found in the model")

    def compute_heatmap(self, img_array: np.ndarray, class_idx: Optional[int] = None):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))
            class_channel = predictions[:, class_idx]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        conv_outputs = conv_outputs[0].numpy()

        for index, weight in enumerate(pooled_grads):
            conv_outputs[:, :, index] *= weight

        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8
        return heatmap, predictions.numpy()

    @staticmethod
    def overlay_heatmap(heatmap, original_img, alpha: float = 0.45):
        if original_img.ndim == 2:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)
        return cv2.addWeighted(original_img.astype(np.uint8), 1 - alpha, heatmap_colored, alpha, 0)


def list_matching_images(root: Path, label_map: Dict[str, int], max_images: int) -> List[Path]:
    candidates = list(root.rglob("*.jpg")) + list(root.rglob("*.jpeg")) + list(root.rglob("*.png"))
    selected: List[Path] = []
    for path in candidates:
        if path.stem in label_map:
            selected.append(path)
            if max_images > 0 and len(selected) >= max_images:
                break
    return selected


def load_image_for_model(path: Path, input_channels: int, target_size: Tuple[int, int]):
    if input_channels == 1:
        img = keras_image.load_img(path, target_size=target_size, color_mode="grayscale")
    else:
        img = keras_image.load_img(path, target_size=target_size, color_mode="rgb")
    img_array = keras_image.img_to_array(img)
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32) / 255.0
    return img_array, np.array(img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train local p10 CXR model and generate Grad-CAM")
    parser.add_argument("--images-root", default="p10", help="Local p10 image root")
    parser.add_argument(
        "--chexpert-local",
        default=str(Path("physionet.org") / "mimic-cxr-2.0.0-chexpert.csv.gz"),
        help="Local CheXpert CSV(.gz)",
    )
    parser.add_argument(
        "--split-local",
        default=str(Path("physionet.org") / "mimic-cxr-2.0.0-split.csv.gz"),
        help="Local split CSV(.gz)",
    )
    parser.add_argument("--label-column", default="Pneumonia", help="CheXpert label column")
    parser.add_argument("--img-size", type=int, default=224, help="Training image size")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max images to load; use 0 to load all matched images from the local p10 tree",
    )
    parser.add_argument("--output-model", default="p10_cxr_model.h5", help="Path to save the trained model")
    parser.add_argument(
        "--gradcam-output-dir",
        default="gradcam_outputs",
        help="Directory for Grad-CAM PNGs",
    )
    parser.add_argument("--gradcam-samples", type=int, default=3, help="Number of local images to explain")
    args = parser.parse_args()

    if not TF_AVAILABLE or tf is None:
        raise RuntimeError(
            "TensorFlow is required for Grad-CAM training. Use the Python 3.11 env with TensorFlow installed."
        ) from TF_IMPORT_ERROR

    images_root = Path(args.images_root)
    chexpert_local = Path(args.chexpert_local)
    split_local = Path(args.split_local)

    if not images_root.exists():
        raise FileNotFoundError(f"Image root not found: {images_root}")
    if not chexpert_local.exists():
        raise FileNotFoundError(f"CheXpert CSV not found: {chexpert_local}")
    if not split_local.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_local}")

    print("Loading local CheXpert metadata...")
    dicom_label_map = build_dicom_label_map_local(
        chexpert_local=str(chexpert_local),
        split_local=str(split_local),
        label_column=args.label_column,
        uncertain_policy="ones",
    )
    print(f"Loaded labels for {len(dicom_label_map):,} DICOM IDs")

    print(f"Loading images from {images_root}...")
    x_data, y_data = load_batch_local(
        local_root=str(images_root),
        img_size=args.img_size,
        dicom_label_map=dicom_label_map,
        limit=args.limit,
    )

    if len(x_data) < 10:
        raise RuntimeError(f"Not enough labeled images loaded from {images_root}. Loaded: {len(x_data)}")

    print(f"Loaded {len(x_data):,} images for training")
    split_index = int(0.8 * len(x_data))
    x_train, x_test = x_data[:split_index], x_data[split_index:]
    y_train, y_test = y_data[:split_index], y_data[split_index:]

    if len(x_test) == 0:
        raise RuntimeError("No test samples after split. Increase the loaded image count.")

    model = build_model(args.img_size)
    print("Training CXR classifier...")
    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    output_model = Path(args.output_model)
    model.save(str(output_model))
    print(f"Saved model to {output_model}")

    sample_paths = list_matching_images(images_root, dicom_label_map, max_images=args.gradcam_samples)
    if not sample_paths:
        print("No matching images found for Grad-CAM visualization.")
        return

    output_dir = Path(args.gradcam_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gradcam = GradCAM(model)
    input_channels = int(model.input_shape[-1])
    target_size = (int(model.input_shape[1]), int(model.input_shape[2]))

    for path in sample_paths:
        img_array, original_img = load_image_for_model(path, input_channels, target_size)
        prediction = float(model.predict(img_array, verbose=0)[0][0])
        heatmap, _ = gradcam.compute_heatmap(img_array, class_idx=0)
        overlay = gradcam.overlay_heatmap(heatmap, original_img, alpha=0.5)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_img, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay | P(pos)={prediction:.2f}")
        axes[2].axis("off")
        plt.tight_layout()

        output_path = output_dir / f"gradcam_{path.stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Grad-CAM visualization: {output_path}")


if __name__ == "__main__":
    main()