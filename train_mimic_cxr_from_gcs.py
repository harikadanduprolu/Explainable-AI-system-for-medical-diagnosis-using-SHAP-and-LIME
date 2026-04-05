import argparse
import importlib
import os
import random
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import storage

joblib = importlib.import_module("joblib")
sklearn_linear_model = importlib.import_module("sklearn.linear_model")
sklearn_metrics = importlib.import_module("sklearn.metrics")
sklearn_pipeline = importlib.import_module("sklearn.pipeline")
sklearn_preprocessing = importlib.import_module("sklearn.preprocessing")

try:
    tf = importlib.import_module("tensorflow")
    layers = importlib.import_module("tensorflow.keras.layers")
    models = importlib.import_module("tensorflow.keras.models")
    TF_AVAILABLE = True
except Exception:
    tf = None
    layers = None
    models = None
    TF_AVAILABLE = False

# -------------------------------
# CONFIG
# -------------------------------
BUCKET_NAME = "mimic-cxr-jpg-2.1.0.physionet.org"
PREFIX = "files/p10"
METADATA_BUCKET = "mimic-cxr-jpg-2.1.0.physionet.org"
CHEXPERT_BLOB = "mimic-cxr-2.0.0-chexpert.csv.gz"
SPLIT_BLOB = "mimic-cxr-2.0.0-split.csv.gz"
IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 42



random.seed(SEED)
np.random.seed(SEED)
if TF_AVAILABLE:
    tf.random.set_seed(SEED)


# -------------------------------
# CONNECT TO GCS
# -------------------------------
def get_bucket(
    bucket_name: str,
    gcp_project: Optional[str] = None,
    use_anonymous_gcs: bool = False,
) -> storage.Bucket:
    if use_anonymous_gcs:
        client = storage.Client.create_anonymous_client()
        return client.bucket(bucket_name)

    project = gcp_project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    if not project:
        raise ValueError(
            "GCP project ID is required for Requester Pays buckets. "
            "Pass --gcp-project or set GOOGLE_CLOUD_PROJECT."
        )
    client = storage.Client(project=project)
    # Required by Requester Pays buckets (like MIMIC-CXR-JPG): bill reads to this project.
    bucket = client.bucket(bucket_name, user_project=project)
    return bucket


def load_csv_from_gcs(bucket: storage.Bucket, blob_name: str) -> pd.DataFrame:
    blob = bucket.blob(blob_name)
    try:
        data = blob.download_as_bytes()
    except NotFound as exc:
        raise FileNotFoundError(f"Blob not found: gs://{bucket.name}/{blob_name}") from exc

    if blob_name.endswith(".gz"):
        return pd.read_csv(BytesIO(data), compression="gzip")
    return pd.read_csv(BytesIO(data))


def load_csv_from_path(path: str) -> pd.DataFrame:
    if path.endswith(".gz"):
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def build_dicom_label_map(
    metadata_bucket: storage.Bucket,
    chexpert_blob: str,
    split_blob: Optional[str],
    label_column: str,
    uncertain_policy: str,
    chexpert_local: Optional[str] = None,
    split_local: Optional[str] = None,
) -> Dict[str, int]:
    if chexpert_local:
        chexpert_df = load_csv_from_path(chexpert_local)
    else:
        chexpert_df = load_csv_from_gcs(metadata_bucket, chexpert_blob)

    if label_column not in chexpert_df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in {chexpert_blob}. "
            f"Available columns: {list(chexpert_df.columns)}"
        )

    if "dicom_id" in chexpert_df.columns:
        merged = chexpert_df.copy()
    else:
        if not split_blob:
            raise ValueError(
                "CheXpert CSV does not contain dicom_id. Provide --split-blob for mapping."
            )
        if split_local:
            split_df = load_csv_from_path(split_local)
        else:
            split_df = load_csv_from_gcs(metadata_bucket, split_blob)
        required_cols = {"subject_id", "study_id", "dicom_id"}
        if not required_cols.issubset(set(split_df.columns)):
            raise ValueError(
                f"Split CSV missing required columns {required_cols}. "
                f"Found: {set(split_df.columns)}"
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
        # drop keeps uncertain labels as missing and removes them below
        label_series = label_series.replace(-1, np.nan)

    valid = merged.assign(_label=label_series).dropna(subset=["dicom_id", "_label"])
    valid = valid[valid["_label"].isin([0, 1])]

    dicom_to_label = {
        str(dicom_id): int(label)
        for dicom_id, label in zip(valid["dicom_id"], valid["_label"])
    }
    if not dicom_to_label:
        raise RuntimeError(
            "No valid labels after processing. Check label column and uncertain policy."
        )
    return dicom_to_label


# -------------------------------
# LOAD IMAGE FROM CLOUD
# -------------------------------
def load_image(blob: storage.Blob, img_size: int) -> Optional[np.ndarray]:
    try:
        img_bytes = blob.download_as_bytes()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        return img
    except Exception:
        return None


# -------------------------------
# LOAD BATCH OF IMAGES
# -------------------------------
def load_batch(
    bucket: storage.Bucket,
    prefix: str,
    img_size: int,
    dicom_label_map: Dict[str, int],
    limit: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    blobs = [
        blob for blob in bucket.list_blobs(prefix=prefix)
        if blob.name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    random.shuffle(blobs)

    images = []
    labels = []

    for blob in blobs:
        dicom_id = blob.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if dicom_id not in dicom_label_map:
            continue

        img = load_image(blob, img_size)
        if img is not None:
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


def load_batch_local(
    local_root: str,
    img_size: int,
    dicom_label_map: Dict[str, int],
    limit: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    root = Path(local_root)
    if not root.exists():
        raise FileNotFoundError(f"Local image root not found: {local_root}")

    candidates = list(root.rglob("*.jpg")) + list(root.rglob("*.jpeg")) + list(root.rglob("*.png"))
    random.shuffle(candidates)

    images = []
    labels = []

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


# -------------------------------
# BUILD MODEL
# -------------------------------
def build_model(img_size: int):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is unavailable in this environment")
    model = models.Sequential([
        layers.Input(shape=(img_size, img_size, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_sklearn_model():
    return sklearn_pipeline.Pipeline([
        ("scaler", sklearn_preprocessing.StandardScaler()),
        (
            "clf",
            sklearn_linear_model.LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
    ])


def prepare_sklearn_features(x_data: np.ndarray, target_size: int = 64) -> np.ndarray:
    flattened = []
    for image in x_data:
        resized = cv2.resize(image.squeeze(), (target_size, target_size))
        flattened.append(resized.astype(np.float32).reshape(-1))
    return np.array(flattened, dtype=np.float32)


# -------------------------------
# MAIN TRAINING
# -------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple CNN on MIMIC-CXR images from GCS")
    parser.add_argument("--bucket", default=BUCKET_NAME, help="GCS bucket name")
    parser.add_argument("--prefix", default=PREFIX, help="Blob prefix within bucket")
    parser.add_argument(
        "--metadata-bucket",
        default=METADATA_BUCKET,
        help="Bucket containing MIMIC-CXR metadata CSVs",
    )
    parser.add_argument(
        "--chexpert-blob",
        default=CHEXPERT_BLOB,
        help="CheXpert labels CSV blob path in metadata bucket",
    )
    parser.add_argument(
        "--split-blob",
        default=SPLIT_BLOB,
        help="Split CSV blob path in metadata bucket (used to map dicom_id)",
    )
    parser.add_argument(
        "--chexpert-local",
        default=None,
        help="Local path to CheXpert labels CSV(.gz); bypasses --chexpert-blob when provided",
    )
    parser.add_argument(
        "--split-local",
        default=None,
        help="Local path to split CSV(.gz); bypasses --split-blob when provided",
    )
    parser.add_argument(
        "--label-column",
        default="Pneumonia",
        help="CheXpert label column to train on",
    )
    parser.add_argument(
        "--uncertain-policy",
        choices=["ones", "zeros", "drop"],
        default="ones",
        help="How to treat uncertain (-1) labels",
    )
    parser.add_argument(
        "--gcp-project",
        default=None,
        help="Google Cloud project ID (required when ADC has no default project)",
    )
    parser.add_argument(
        "--use-anonymous-gcs",
        action="store_true",
        help="Use anonymous GCS client for publicly readable buckets",
    )
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="Square image size")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max number of images to load; use 0 to load all matched images",
    )
    parser.add_argument("--output", default="mimic_model.h5", help="Path to save model")
    parser.add_argument(
        "--images-local-root",
        default=None,
        help="Local root folder containing downloaded CXR JPG files; when provided, skips GCS image reads",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "tensorflow", "sklearn"],
        default="auto",
        help="Training backend. Use sklearn to run on Python 3.13 without TensorFlow.",
    )
    args = parser.parse_args()

    backend = args.backend
    if backend == "auto":
        backend = "tensorflow" if TF_AVAILABLE else "sklearn"
    if backend == "tensorflow" and not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow backend requested, but TensorFlow is not available in this environment. "
            "Use --backend sklearn or run under Python 3.11."
        )

    print("Loading metadata from cloud...")
    metadata_bucket = get_bucket(
        args.metadata_bucket,
        gcp_project=args.gcp_project,
        use_anonymous_gcs=args.use_anonymous_gcs,
    )
    dicom_label_map = build_dicom_label_map(
        metadata_bucket=metadata_bucket,
        chexpert_blob=args.chexpert_blob,
        split_blob=args.split_blob,
        label_column=args.label_column,
        uncertain_policy=args.uncertain_policy,
        chexpert_local=args.chexpert_local,
        split_local=args.split_local,
    )
    print(f"Loaded labels for {len(dicom_label_map)} DICOM images")

    if args.images_local_root:
        print(f"Loading images from local folder: {args.images_local_root}")
        x_data, y_data = load_batch_local(
            local_root=args.images_local_root,
            img_size=args.img_size,
            dicom_label_map=dicom_label_map,
            limit=args.limit,
        )
    else:
        print("Loading images from cloud...")
        image_bucket = get_bucket(
            args.bucket,
            gcp_project=args.gcp_project,
            use_anonymous_gcs=args.use_anonymous_gcs,
        )
        x_data, y_data = load_batch(
            image_bucket,
            args.prefix,
            args.img_size,
            dicom_label_map,
            limit=args.limit,
        )

    if len(x_data) < 10:
        raise RuntimeError(
            f"Not enough images loaded from the selected source. Loaded: {len(x_data)}"
        )

    pos_rate = float(np.mean(y_data)) if len(y_data) else 0.0
    print(f"Loaded {len(x_data)} labeled images (positive rate: {pos_rate:.3f})")

    split = int(0.8 * len(x_data))
    x_train, x_test = x_data[:split], x_data[split:]
    y_train, y_test = y_data[:split], y_data[split:]

    if len(x_test) == 0:
        raise RuntimeError("No test samples after split. Increase --limit.")

    output_path = Path(args.output)

    if backend == "tensorflow":
        model = build_model(args.img_size)

        print("Training TensorFlow CNN...")
        model.fit(
            x_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(x_test, y_test),
            verbose=1,
        )

        print("Evaluating...")
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {acc:.4f}")

        model.save(str(output_path))
        print(f"Model saved to: {output_path}")
        return

    print("Training sklearn fallback model for Python 3.13...")
    x_train_flat = prepare_sklearn_features(x_train, target_size=64)
    x_test_flat = prepare_sklearn_features(x_test, target_size=64)

    model = build_sklearn_model()
    model.fit(x_train_flat, y_train)

    probabilities = model.predict_proba(x_test_flat)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    acc = sklearn_metrics.accuracy_score(y_test, predictions)
    loss = sklearn_metrics.log_loss(y_test, probabilities, labels=[0, 1])

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    if output_path.suffix.lower() not in {".joblib", ".pkl"}:
        output_path = output_path.with_suffix(".joblib")
    joblib.dump(model, output_path)
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
