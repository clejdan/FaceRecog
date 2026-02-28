import base64
import requests

def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

payload = {
    "suspect_id": "cowboy_1",
    "frame_b64": img_to_b64("test.png"),   # full frame — Jetson B detects face
}

r = requests.post("http://127.0.0.1:9002/event", json=payload)
data = r.json()

with open("poster_out.png", "wb") as f:
    f.write(base64.b64decode(data["poster_image"]))

print("Bounty:", data["bounty"])
print("Poster saved to poster_out.png")
