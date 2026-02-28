from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import logging
import os

import cv2
import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("jetson_b")

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(__file__)
ASSETS_DIR    = os.path.join(BASE_DIR, "assets")
MUSTACHE_PATH = os.path.join(ASSETS_DIR, "mustache.png")

FONT_BOLD  = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

# ── Haar cascade ──────────────────────────────────────────────────────────────

try:
    _HAAR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
except AttributeError:
    _HAAR_XML = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

_face_cascade = cv2.CascadeClassifier(_HAAR_XML)

# ── Poster geometry (px) ──────────────────────────────────────────────────────

POSTER_W   = 800
POSTER_H   = 1000
FACE_BOX_W = 480
FACE_BOX_H = 480
FACE_TOP_Y = 280
FACE_X     = (POSTER_W - FACE_BOX_W) // 2   # = 160

FACE_PADDING = 40

# ── Colour palette ────────────────────────────────────────────────────────────

PARCHMENT_BG   = (235, 215, 175, 255)
BORDER_COLOR   = (80,  45,  15,  255)
TEXT_DARK      = (40,  20,   5,  255)
TEXT_RED       = (160, 20,  10,  255)
ORNAMENT_COLOR = (100, 60,  20,  255)

MUSTACHE_BG_THRESHOLD = 180   # RGB channels above this → background (transparent)

# ── In-memory stores ──────────────────────────────────────────────────────────

BOUNTIES:     dict = {}   # suspect_id -> int
POSTERS:      dict = {}   # suspect_id -> base64 PNG str
POSTER_ORDER: list = []   # insertion order

# ── Mustache asset (loaded once at startup) ───────────────────────────────────

def _load_mustache_rgba() -> Image.Image:
    """Load mustache.png and make its near-white background transparent."""
    img = Image.open(MUSTACHE_PATH).convert("RGBA")
    arr = np.array(img)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mask = (r > MUSTACHE_BG_THRESHOLD) & (g > MUSTACHE_BG_THRESHOLD) & (b > MUSTACHE_BG_THRESHOLD)
    arr[mask, 3] = 0
    return Image.fromarray(arr, "RGBA")

MUSTACHE_RGBA = _load_mustache_rgba()

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI()

class Event(BaseModel):
    suspect_id: str
    frame_b64:  str   # full captured frame, PNG-encoded base64

# ── Image conversion helpers ──────────────────────────────────────────────────

def b64_to_cv2(b64_str: str) -> np.ndarray:
    data = base64.b64decode(b64_str)
    arr  = np.frombuffer(data, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed — invalid image bytes")
    return img

def cv2_to_pil_rgba(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, "RGB").convert("RGBA")

def image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ── Face detection ────────────────────────────────────────────────────────────

def detect_and_crop_face(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Detect the largest face in frame_bgr, return a padded BGR crop.
    Falls back to a center-square crop if no face is found.
    """
    fh, fw = frame_bgr.shape[:2]
    gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray, gray)

    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        log.warning("No face detected — using center-square fallback crop")
        side = min(fw, fh) // 2
        cx, cy = fw // 2, fh // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        return frame_bgr[y1 : y1 + side, x1 : x1 + side]

    # Select the largest face (most prominent suspect)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    log.info("Face detected at (%d, %d) size %dx%d", x, y, w, h)

    x1 = max(0, x - FACE_PADDING)
    y1 = max(0, y - FACE_PADDING)
    x2 = min(fw, x + w + FACE_PADDING)
    y2 = min(fh, y + h + FACE_PADDING)
    return frame_bgr[y1:y2, x1:x2]

# ── Poster drawing helpers ────────────────────────────────────────────────────

def _draw_centered(draw: ImageDraw.ImageDraw, text: str,
                   font: ImageFont.FreeTypeFont, y: int,
                   fill: tuple, shadow: int = 0) -> None:
    bbox   = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    x      = (POSTER_W - text_w) // 2
    if shadow:
        draw.text((x + shadow, y + shadow), text, font=font, fill=(0, 0, 0, 80))
    draw.text((x, y), text, font=font, fill=fill)

def _ornament_line(draw: ImageDraw.ImageDraw, y: int) -> None:
    m = 45
    draw.line([(m, y),     (POSTER_W - m, y)],     fill=ORNAMENT_COLOR, width=1)
    draw.line([(m, y + 3), (POSTER_W - m, y + 3)], fill=ORNAMENT_COLOR, width=3)
    draw.line([(m, y + 7), (POSTER_W - m, y + 7)], fill=ORNAMENT_COLOR, width=1)

def _apply_mustache(poster: Image.Image) -> None:
    """Paste scaled mustache onto the face zone."""
    target_w  = int(FACE_BOX_W * 0.55)
    scale     = target_w / MUSTACHE_RGBA.width
    target_h  = int(MUSTACHE_RGBA.height * scale)
    scaled    = MUSTACHE_RGBA.resize((target_w, target_h), Image.LANCZOS)
    mx = FACE_X + (FACE_BOX_W - target_w) // 2
    my = FACE_TOP_Y + int(FACE_BOX_H * 0.62)
    poster.paste(scaled, (mx, my), scaled)

# ── Poster compositor ─────────────────────────────────────────────────────────

def make_poster(face_bgr: np.ndarray, suspect_id: str, bounty: int) -> Image.Image:
    # ── Canvas with parchment background ─────────────────────────────────────
    poster = Image.new("RGBA", (POSTER_W, POSTER_H), PARCHMENT_BG)

    # ── Noise/grain texture layer ─────────────────────────────────────────────
    noise       = np.random.randint(0, 30, (POSTER_H, POSTER_W, 4), dtype=np.uint8)
    noise[:, :, 3] = 38   # ~15% opacity
    poster      = Image.alpha_composite(poster, Image.fromarray(noise, "RGBA"))
    draw        = ImageDraw.Draw(poster)

    # ── Double-line border ────────────────────────────────────────────────────
    draw.rectangle([10, 10, POSTER_W - 10, POSTER_H - 10], outline=BORDER_COLOR, width=6)
    draw.rectangle([22, 22, POSTER_W - 22, POSTER_H - 22], outline=BORDER_COLOR, width=2)
    for cx, cy in [(10, 10), (POSTER_W - 10, 10), (10, POSTER_H - 10), (POSTER_W - 10, POSTER_H - 10)]:
        draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=BORDER_COLOR)

    # ── Load fonts ────────────────────────────────────────────────────────────
    f_wanted   = ImageFont.truetype(FONT_BOLD,  110)
    f_subtitle = ImageFont.truetype(FONT_BOLD,   38)
    f_bounty   = ImageFont.truetype(FONT_BOLD,   52)
    f_small    = ImageFont.truetype(FONT_SERIF,  15)

    # ── "WANTED" ──────────────────────────────────────────────────────────────
    _draw_centered(draw, "WANTED", f_wanted, y=45, fill=TEXT_DARK, shadow=3)

    # ── "DEAD OR ALIVE" ───────────────────────────────────────────────────────
    _draw_centered(draw, "DEAD OR ALIVE", f_subtitle, y=158, fill=TEXT_RED)

    # ── Ornament lines ────────────────────────────────────────────────────────
    _ornament_line(draw, y=205)
    _ornament_line(draw, y=790)
    _ornament_line(draw, y=900)

    # ── Face photo ────────────────────────────────────────────────────────────
    face_pil = cv2_to_pil_rgba(face_bgr).resize((FACE_BOX_W, FACE_BOX_H), Image.LANCZOS)
    draw.rectangle(
        [FACE_X - 4, FACE_TOP_Y - 4, FACE_X + FACE_BOX_W + 4, FACE_TOP_Y + FACE_BOX_H + 4],
        outline=BORDER_COLOR, width=3,
    )
    poster.paste(face_pil, (FACE_X, FACE_TOP_Y), face_pil)

    # ── Mustache ──────────────────────────────────────────────────────────────
    _apply_mustache(poster)

    # ── Bounty text ───────────────────────────────────────────────────────────
    draw = ImageDraw.Draw(poster)   # re-bind after paste operations
    _draw_centered(draw, f"Bounty: ${bounty:,}", f_bounty, y=825, fill=TEXT_RED, shadow=2)

    # ── Footer ────────────────────────────────────────────────────────────────
    _draw_centered(draw, "Wanted by the Marshal's Office", f_small, y=915, fill=ORNAMENT_COLOR)
    _draw_centered(draw, f"[{suspect_id}]", f_small, y=940, fill=ORNAMENT_COLOR)

    return poster

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/event")
def handle_event(event: Event):
    bounty = BOUNTIES.get(event.suspect_id, 0) + 100
    BOUNTIES[event.suspect_id] = bounty

    frame_bgr  = b64_to_cv2(event.frame_b64)
    face_bgr   = detect_and_crop_face(frame_bgr)
    poster_img = make_poster(face_bgr, event.suspect_id, bounty)
    poster_b64 = image_to_b64(poster_img)

    POSTERS[event.suspect_id] = poster_b64
    if event.suspect_id not in POSTER_ORDER:
        POSTER_ORDER.append(event.suspect_id)

    log.info("Poster generated — suspect: %s  bounty: $%d", event.suspect_id, bounty)

    return {
        "suspect_id":   event.suspect_id,
        "bounty":       bounty,
        "poster_image": poster_b64,
    }


@app.get("/board", response_class=HTMLResponse)
def bounty_board():
    if not POSTER_ORDER:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
          <meta http-equiv="refresh" content="5">
          <meta charset="utf-8">
          <title>Bounty Board</title>
          <style>
            body { background:#1c0f04; color:#f0d080;
                   font-family:'Georgia',serif; text-align:center; }
            h1   { font-size:3em; margin-top:120px; letter-spacing:0.2em; }
            p    { color:#804020; margin-top:20px; letter-spacing:0.1em; }
          </style>
        </head>
        <body>
          <h1>NO OUTLAWS YET</h1>
          <p>The Marshal is watching...</p>
        </body>
        </html>
        """)

    cards = ""
    for sid in reversed(POSTER_ORDER):
        b64    = POSTERS[sid]
        bounty = BOUNTIES[sid]
        cards += f"""
        <div class="card">
          <img src="data:image/png;base64,{b64}" alt="{sid}">
          <div class="label">{sid}<br>${bounty:,}</div>
        </div>
        """

    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta http-equiv="refresh" content="5">
      <meta charset="utf-8">
      <title>Wild West Bounty Board</title>
      <style>
        *      {{ box-sizing:border-box; margin:0; padding:0; }}
        body   {{
          background:#1c0f04;
          background-image: repeating-linear-gradient(
            45deg,rgba(255,200,100,.03) 0,rgba(255,200,100,.03) 1px,
            transparent 1px,transparent 10px);
          font-family:'Georgia',serif; color:#f0d080; min-height:100vh;
        }}
        header {{
          text-align:center; padding:28px 20px 18px;
          border-bottom:3px solid #7a4a10; background:#2b1508;
        }}
        header h1 {{
          font-size:3.2em; letter-spacing:0.25em;
          text-shadow:3px 3px 8px #000; color:#e8c040;
        }}
        header p  {{ font-size:1em; color:#b08840; margin-top:6px; letter-spacing:0.1em; }}
        .board    {{
          display:flex; flex-wrap:wrap; justify-content:center;
          gap:28px; padding:36px 20px; max-width:1400px; margin:0 auto;
        }}
        .card     {{
          background:#2b1508; border:3px solid #7a4a10; border-radius:4px;
          padding:10px; width:255px; box-shadow:6px 6px 20px rgba(0,0,0,.8);
          transition:transform .2s;
        }}
        .card:hover {{ transform:scale(1.04); }}
        .card img   {{ width:100%; display:block; border:2px solid #7a4a10; }}
        .label      {{
          text-align:center; margin-top:9px; font-size:.95em;
          color:#e8c040; letter-spacing:.05em; line-height:1.5;
        }}
        footer {{
          text-align:center; padding:18px; color:#604020;
          font-size:.78em; border-top:1px solid #3a2008;
        }}
      </style>
    </head>
    <body>
      <header>
        <h1>BOUNTY BOARD</h1>
        <p>Wild West Theft Detection — Marshal's Office</p>
      </header>
      <div class="board">{cards}</div>
      <footer>
        Auto-refreshes every 5 seconds &nbsp;|&nbsp;
        {len(POSTER_ORDER)} outlaw(s) on record
      </footer>
    </body>
    </html>
    """)
