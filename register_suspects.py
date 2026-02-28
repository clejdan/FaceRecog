#!/usr/bin/env python3
"""
register_suspects.py
One-time setup: reads photos from CJ/, Cameron/, Tolga/ folders,
computes a mean SFace embedding per person, saves to suspects.pkl.

Usage:
  python3 register_suspects.py
"""

import glob
import os
import pickle

import cv2
import numpy as np

YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"
OUTPUT_FILE = "suspects.pkl"

# Folder name → suspect_id used as the key sent in /event
SUSPECT_DIRS = {
    "CJ":      "CJ",
    "Cameron": "Cameron",
    "Tolga":   "Tolga",
}

SCORE_THRESHOLD = 0.6   # YuNet detection confidence threshold


def load_detector(w: int, h: int) -> cv2.FaceDetectorYN:
    det = cv2.FaceDetectorYN.create(
        model=YUNET_MODEL,
        config="",
        input_size=(w, h),
        score_threshold=SCORE_THRESHOLD,
        nms_threshold=0.3,
        top_k=5000,
    )
    return det


def compute_embedding(
    recognizer: cv2.FaceRecognizerSF,
    detector: cv2.FaceDetectorYN,
    img_bgr: np.ndarray,
) -> np.ndarray | None:
    h, w = img_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img_bgr)

    if faces is None or len(faces) == 0:
        return None

    # Use the highest-confidence detection
    best = faces[np.argmax(faces[:, 14])]
    aligned = recognizer.alignCrop(img_bgr, best)
    feat = recognizer.feature(aligned)
    return feat.flatten()


def process_folder(
    folder: str,
    recognizer: cv2.FaceRecognizerSF,
    detector: cv2.FaceDetectorYN,
) -> np.ndarray | None:
    extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(folder, ext)))

    if not paths:
        print(f"  [!] No images found in {folder}/")
        return None

    embeddings = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [!] Could not read {path}, skipping")
            continue
        emb = compute_embedding(recognizer, detector, img)
        if emb is None:
            print(f"  [!] No face detected in {os.path.basename(path)}, skipping")
        else:
            embeddings.append(emb)

    if not embeddings:
        print(f"  [!] No usable faces found in {folder}/")
        return None

    mean_emb = np.mean(embeddings, axis=0)
    # L2-normalize the mean embedding once
    mean_emb /= np.linalg.norm(mean_emb)
    print(f"  OK — {len(embeddings)}/{len(paths)} photos used")
    return mean_emb


def main() -> None:
    recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
    # Detector is re-sized per image; initial size doesn't matter much
    detector = load_detector(640, 640)

    suspects: dict[str, np.ndarray] = {}

    for folder, suspect_id in SUSPECT_DIRS.items():
        print(f"\nProcessing {folder}/ → '{suspect_id}'")
        if not os.path.isdir(folder):
            print(f"  [!] Directory not found: {folder}/")
            continue
        emb = process_folder(folder, recognizer, detector)
        if emb is not None:
            suspects[suspect_id] = emb

    if not suspects:
        print("\n[ERROR] No suspect embeddings registered. Check folder paths.")
        return

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(suspects, f)

    print(f"\nSaved {len(suspects)} suspect(s) to {OUTPUT_FILE}:")
    for name in suspects:
        print(f"  {name}")


if __name__ == "__main__":
    main()
