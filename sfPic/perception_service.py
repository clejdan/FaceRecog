#!/usr/bin/env python3
"""
perception_service.py — Jetson A
Wild West Theft Detection Demo

Responsibilities:
  - Camera ingest (USB or GStreamer CSI)
  - ROI-based box presence detection via pixel diff against reference
  - Haar cascade face detection + nearest-suspect selection
  - Non-blocking event dispatch to Jetson B (POST /event)

Usage:
  python3 perception_service.py --jetson-b http://192.168.1.XXX:9002

Keys (when display enabled):
  q — quit
  r — recalibrate (keep box in frame, press r)
  s — save current reference ROI snapshot to ref_debug.png
"""

import argparse
import base64
import logging
import os
import pickle
import threading
import time

import cv2
import numpy as np
import requests

# ── CLI ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Jetson A perception service")
parser.add_argument(
    "--jetson-b",
    default=os.environ.get("JETSON_B_URL", "http://192.168.1.100:9002"),
    help="Jetson B base URL (env: JETSON_B_URL)",
)
parser.add_argument(
    "--camera",
    default="0",
    help="Camera index (int) or GStreamer pipeline string",
)
parser.add_argument("--width",  type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps",    type=int, default=10, help="Target capture FPS")
parser.add_argument(
    "--roi",
    nargs=4, type=int, metavar=("X", "Y", "W", "H"),
    default=[300, 200, 300, 250],
    help="Box ROI: x y width height in pixels",
)
parser.add_argument(
    "--cal-frames",
    type=int, default=30,
    help="Number of frames averaged to build reference (box must be present)",
)
parser.add_argument(
    "--diff-threshold",
    type=float, default=25.0,
    help="Mean absolute pixel diff above which box is considered removed",
)
parser.add_argument(
    "--debounce",
    type=int, default=3,
    help="Consecutive above-threshold frames required before triggering",
)
parser.add_argument(
    "--cooldown",
    type=float, default=15.0,
    help="Seconds to wait between event dispatches",
)
parser.add_argument(
    "--no-display",
    action="store_true",
    help="Headless mode — disable OpenCV window",
)
args = parser.parse_args()

JETSON_B_EVENT_URL = args.jetson_b.rstrip("/") + "/event"
_BASE              = os.path.dirname(os.path.abspath(__file__))
YUNET_MODEL        = os.path.join(_BASE, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL        = os.path.join(_BASE, "face_recognition_sface_2021dec.onnx")
SUSPECTS_FILE      = os.path.join(_BASE, "suspects.pkl")
MATCH_THRESHOLD    = 0.35   # cosine similarity; ≥ threshold → recognized
SUSPECT_PADDING    = 50     # px added around face bbox for fallback crop

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("jetson_a")

# ── Load face recognizer and suspects ────────────────────────────────────────

_yunet      = cv2.FaceDetectorYN.create(
    YUNET_MODEL, "", (640, 640), score_threshold=0.6, nms_threshold=0.3
)
_recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

if not os.path.exists(SUSPECTS_FILE):
    log.error("suspects.pkl not found — run register_suspects.py first")
    raise SystemExit(1)

with open(SUSPECTS_FILE, "rb") as _f:
    SUSPECTS: dict = pickle.load(_f)

log.info("Loaded %d suspect embeddings: %s", len(SUSPECTS), list(SUSPECTS.keys()))

# ── Phase constants ──────────────────────────────────────────────────────────

CALIBRATING = "CALIBRATING"
BOX_PRESENT = "BOX_PRESENT"
BOX_REMOVED = "BOX_REMOVED"

# ── Mutable state ────────────────────────────────────────────────────────────

phase           = CALIBRATING
cal_buffer      = []
ref_roi         = None          # uint8 numpy array (h, w, 3)
debounce_count  = 0             # consecutive above-threshold frames
last_event_time = 0.0
recal_requested = False         # set by keypress handler

# ── Camera ───────────────────────────────────────────────────────────────────

def open_camera(src: str) -> cv2.VideoCapture:
    try:
        idx = int(src)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          args.fps)
    return cap

# ── Vision helpers ───────────────────────────────────────────────────────────

def crop_roi(frame: np.ndarray, roi: tuple) -> np.ndarray:
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def roi_diff(ref: np.ndarray, current: np.ndarray) -> float:
    """Mean absolute difference between grayscale ROI crops."""
    g_ref = cv2.cvtColor(ref,     cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_cur = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(g_ref - g_cur)))


def build_reference(buffer: list) -> np.ndarray:
    """Average a list of BGR frames to produce a stable reference."""
    stack = np.stack(buffer, axis=0).astype(np.float32)
    return np.mean(stack, axis=0).astype(np.uint8)


def identify_suspect(frame: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Detect a face in frame using YuNet, compute its SFace embedding,
    and compare against registered suspects.
    Returns (face_crop_bgr, suspect_id).
    Falls back to a center crop and 'unknown_outlaw' if no face found.
    """
    fh, fw = frame.shape[:2]
    _yunet.setInputSize((fw, fh))
    _, faces = _yunet.detect(frame)

    if faces is None or len(faces) == 0:
        log.warning("No face detected — using center-crop fallback")
        side = min(fw, fh) // 3
        cx, cy = fw // 2, fh // 2
        crop = frame[cy - side // 2: cy + side // 2, cx - side // 2: cx + side // 2]
        return crop, "unknown_outlaw"

    # Best detection = highest confidence
    best_face = faces[np.argmax(faces[:, 14])]

    # Aligned crop for recognition
    aligned = _recognizer.alignCrop(frame, best_face)
    query_emb = _recognizer.feature(aligned).flatten()
    query_emb /= np.linalg.norm(query_emb)

    # Cosine similarity against each registered suspect
    best_name  = "unknown_outlaw"
    best_score = -1.0
    for name, ref_emb in SUSPECTS.items():
        score = float(np.dot(query_emb, ref_emb))
        log.debug("  similarity[%s] = %.4f", name, score)
        if score > best_score:
            best_score = score
            best_name  = name

    if best_score < MATCH_THRESHOLD:
        log.info("No match above threshold (best=%.3f) — unknown_outlaw", best_score)
        best_name = "unknown_outlaw"
    else:
        log.info("Identified as '%s' (cosine=%.3f)", best_name, best_score)

    # Return padded face crop from original frame for the poster
    x, y, w, h = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
    x1 = max(0, x - SUSPECT_PADDING)
    y1 = max(0, y - SUSPECT_PADDING)
    x2 = min(fw, x + w + SUSPECT_PADDING)
    y2 = min(fh, y + h + SUSPECT_PADDING)
    return frame[y1:y2, x1:x2], best_name


# ── Event dispatch ───────────────────────────────────────────────────────────

def encode_crop(crop_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", crop_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def dispatch_event(suspect_id: str, frame_bgr: np.ndarray) -> None:
    """Non-blocking fire-and-forget. Does not delay the capture loop."""
    def _post():
        try:
            b64 = encode_crop(frame_bgr)   # encode_crop works on any BGR array
            r = requests.post(
                JETSON_B_EVENT_URL,
                json={"suspect_id": suspect_id, "frame_b64": b64},
                timeout=10,
            )
            d = r.json()
            log.info(
                "  >> Event ACK — suspect: %s  bounty: $%d",
                d.get("suspect_id"), d.get("bounty", 0),
            )
        except Exception as exc:
            log.error("Event dispatch failed: %s", exc)

    threading.Thread(target=_post, daemon=True).start()


# ── Overlay ──────────────────────────────────────────────────────────────────

PHASE_COLORS = {
    CALIBRATING: (0,  165, 255),   # orange
    BOX_PRESENT: (0,  200,   0),   # green
    BOX_REMOVED: (0,    0, 220),   # red
}


def draw_overlay(frame: np.ndarray, roi: tuple, diff: float) -> None:
    x, y, w, h = roi
    color = PHASE_COLORS.get(phase, (255, 255, 255))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"Phase:     {phase}",          (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"ROI diff:  {diff:.1f}",       (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Threshold: {args.diff_threshold}", (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"Debounce:  {debounce_count}/{args.debounce}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, "r=recal  s=save  q=quit",      (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


# ── Main loop ────────────────────────────────────────────────────────────────

def start_calibration() -> None:
    global phase, cal_buffer, ref_roi, debounce_count
    phase          = CALIBRATING
    cal_buffer     = []
    ref_roi        = None
    debounce_count = 0
    log.info("Calibrating — keep box in place for %d frames...", args.cal_frames)


def main() -> None:
    global phase, cal_buffer, ref_roi, debounce_count
    global last_event_time, recal_requested

    cap = open_camera(args.camera)
    if not cap.isOpened():
        log.error("Cannot open camera: %s", args.camera)
        return

    roi      = tuple(args.roi)
    frame_dt = 1.0 / args.fps

    start_calibration()

    try:
        while True:
            t0 = time.monotonic()

            ok, frame = cap.read()
            if not ok:
                log.warning("Frame read failed — retrying")
                time.sleep(0.05)
                continue

            # Recalibration requested via keypress
            if recal_requested:
                recal_requested = False
                start_calibration()

            current_roi_img = crop_roi(frame, roi)
            diff = 0.0

            # ── CALIBRATING ──────────────────────────────────────────────
            if phase == CALIBRATING:
                cal_buffer.append(current_roi_img.copy())
                n = len(cal_buffer)
                if n % 10 == 0:
                    log.info("  Calibrating %d/%d...", n, args.cal_frames)
                if n >= args.cal_frames:
                    ref_roi = build_reference(cal_buffer)
                    cal_buffer.clear()
                    phase = BOX_PRESENT
                    log.info("Calibration complete. Monitoring for box removal.")

            # ── BOX_PRESENT ──────────────────────────────────────────────
            elif phase == BOX_PRESENT:
                diff = roi_diff(ref_roi, current_roi_img)

                if diff > args.diff_threshold:
                    debounce_count += 1
                    if debounce_count >= args.debounce:
                        face_crop, suspect_id = identify_suspect(frame)
                        log.info(
                            "BOX REMOVED — diff=%.1f  suspect=%s  dispatching event",
                            diff, suspect_id,
                        )
                        dispatch_event(suspect_id, face_crop)
                        last_event_time = time.time()
                        debounce_count  = 0
                        phase           = BOX_REMOVED
                else:
                    debounce_count = 0   # reset on any stable frame

            # ── BOX_REMOVED ──────────────────────────────────────────────
            elif phase == BOX_REMOVED:
                diff    = roi_diff(ref_roi, current_roi_img)
                elapsed = time.time() - last_event_time

                if elapsed >= args.cooldown:
                    if diff <= args.diff_threshold * 0.6:
                        # Box returned to shelf
                        log.info("Box returned — resuming monitoring")
                        phase = BOX_PRESENT
                    else:
                        # Box still gone — repeat offender, re-trigger
                        face_crop, suspect_id = identify_suspect(frame)
                        log.info(
                            "Box still absent post-cooldown — re-triggering  diff=%.1f  suspect=%s",
                            diff, suspect_id,
                        )
                        dispatch_event(suspect_id, face_crop)
                        last_event_time = time.time()

            # ── Display ──────────────────────────────────────────────────
            if not args.no_display:
                draw_overlay(frame, roi, diff)
                cv2.imshow("Jetson A — Perception", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    recal_requested = True
                    log.info("Recalibration requested")
                elif key == ord("s") and ref_roi is not None:
                    cv2.imwrite("ref_debug.png", ref_roi)
                    log.info("Reference ROI saved to ref_debug.png")

            # Pace to target FPS
            sleep_t = frame_dt - (time.monotonic() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        log.info("Shutdown complete")


if __name__ == "__main__":
    main()
