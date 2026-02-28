import cv2

img = cv2.imread("test.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(80, 80)
)

if len(faces) == 0:
    raise RuntimeError("No face detected")

# pick largest face (good for single-person photos)
x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

# optional padding around face
pad = int(0.25 * max(w, h))
x1 = max(0, x - pad)
y1 = max(0, y - pad)
x2 = min(img.shape[1], x + w + pad)
y2 = min(img.shape[0], y + h + pad)

face_crop = img[y1:y2, x1:x2]
cv2.imwrite("face_crop.png", face_crop)
print("Saved face_crop.png")