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
FACE_CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
SUSPECT_PADDING    = 50   # px added around face bbox for crop

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("jetson_a")

# ── Load Haar cascade ────────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if face_cascade.empty():
    log.error("Haar cascade not found: %s", FACE_CASCADE_PATH)
    raise SystemExit(1)

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
suspect_counter = 0
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


def select_suspect(frame: np.ndarray, roi: tuple) -> np.ndarray:
    """
    Detect faces in frame. Return the crop of the face nearest to the
    ROI center, with padding. Falls back to a region adjacent to the ROI
    if no face is detected.
    """
    rx, ry, rw, rh = roi
    roi_cx = rx + rw // 2
    roi_cy = ry + rh // 2
    fh, fw = frame.shape[:2]

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        log.warning("No face detected — using ROI-adjacent fallback crop")
        fx = max(0, roi_cx - 160)
        fy = max(0, ry - 220)
        return frame[fy : min(fh, ry + rh + 100), fx : min(fw, fx + 320)]

    # Manhattan distance from face center to ROI center
    best = min(
        faces,
        key=lambda f: abs((f[0] + f[2] // 2) - roi_cx)
                    + abs((f[1] + f[3] // 2) - roi_cy),
    )
    x, y, w, h = best
    x1 = max(0, x - SUSPECT_PADDING)
    y1 = max(0, y - SUSPECT_PADDING)
    x2 = min(fw, x + w + SUSPECT_PADDING)
    y2 = min(fh, y + h + SUSPECT_PADDING)
    return frame[y1:y2, x1:x2]


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
    global last_event_time, suspect_counter, recal_requested

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
                        suspect_counter += 1
                        suspect_id = f"outlaw_{suspect_counter:04d}"
                        log.info(
                            "BOX REMOVED — diff=%.1f  suspect=%s  dispatching event",
                            diff, suspect_id,
                        )
                        dispatch_event(suspect_id, frame.copy())
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
                        suspect_counter += 1
                        suspect_id = f"outlaw_{suspect_counter:04d}"
                        log.info(
                            "Box still absent post-cooldown — re-triggering  diff=%.1f  suspect=%s",
                            diff, suspect_id,
                        )
                        dispatch_event(suspect_id, frame.copy())
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
