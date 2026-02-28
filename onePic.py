
import sys, pickle, cv2, numpy as np

YUNET  = "face_detection_yunet_2023mar.onnx"
SFACE  = "face_recognition_sface_2021dec.onnx"
THRESH = 0.35

# ── change this to any image path ──────────────────────────────────────────
IMG = "/home/clejah/wildWest/ReceivedImages/img_1772266709.jpg"
# ───────────────────────────────────────────────────────────────────────────
yunet      = cv2.FaceDetectorYN.create(YUNET, "", (640, 640), score_threshold=0.6)
recognizer = cv2.FaceRecognizerSF.create(SFACE, "")

with open("suspects.pkl", "rb") as f:
    suspects = pickle.load(f)

frame = cv2.imread(IMG)
if frame is None:
    print("Could not read image:", IMG); exit(1)

fh, fw = frame.shape[:2]
yunet.setInputSize((fw, fh))
_, faces = yunet.detect(frame)

if faces is None or len(faces) == 0:
    print("No face detected in image")
    exit(0)

best_face  = faces[np.argmax(faces[:, 14])]
aligned    = recognizer.alignCrop(frame, best_face)
query_emb  = recognizer.feature(aligned).flatten()
query_emb /= np.linalg.norm(query_emb)

print(f"\nImage: {IMG}")
print(f"{'Name':<12} {'Score':>8}  {'Match?'}")
print("-" * 35)
for name, ref_emb in suspects.items():
    score = float(np.dot(query_emb, ref_emb))
    match = "<-- MATCH" if score >= THRESH else ""
    print(f"{name:<12} {score:>8.4f}  {match}")