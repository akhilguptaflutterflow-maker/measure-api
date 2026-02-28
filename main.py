from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np

app = FastAPI()

# ---------- INPUT MODEL ----------
class ImageData(BaseModel):
    image: str
    garment_type: str


# ---------- BASE64 DECODE ----------
def decode_image(base64_str):
    header_removed = base64_str.split(",")[-1]
    img_bytes = base64.b64decode(header_removed)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


# ---------- BASE64 ENCODE ----------
def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    base64_img = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_img}"


# ---------- CARD DETECTION ----------
def find_card(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,50,150)

    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_ratio_diff = 999

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        ratio = w/h if h>0 else 0
        diff = abs(ratio - 1.586)

        # FILTER SIZE + RATIO
        if diff < 0.3 and 200 < w < 600 and 100 < h < 400:
            best = (x,y,w,h)
            best_ratio_diff = diff

    return best


# ---------- CLOTH DETECTION ----------
def find_cloth(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)[1]

    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    big_contours = [c for c in contours if cv2.contourArea(c) > 50000]

    largest = max(big_contours, key=cv2.contourArea)

    x,y,w,h = cv2.boundingRect(largest)
    return (x,y,w,h)


# ---------- DRAW OVERLAY ----------
def draw_overlay(image, card_box, cloth_box):
    img = image.copy()

    # CARD
    if card_box is not None:
        x,y,w,h = card_box
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
        cv2.putText(img, "CARD", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # CLOTH
    if cloth_box is not None:
        x,y,w,h = cloth_box
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 3)
        cv2.putText(img, "CLOTH", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    return img


# ---------- API ----------
@app.post("/measure")
def measure(data: ImageData):

    img = decode_image(data.image)

    # detect
    card = find_card(img)
    cloth = find_cloth(img)

    if card is None:
        return {"error": "Card not detected"}

    # ---------- SCALE ----------
    x,y,w,h = card
    pixel_per_inch = w / 3.37

    # ---------- CLOTH SIZE ----------
    cx,cy,cw,ch = cloth

    width_in = cw / pixel_per_inch
    height_in = ch / pixel_per_inch

    size_width = width_in * 2
    size_height = height_in * 2

    # ---------- OVERLAY IMAGE ----------
    overlay_img = draw_overlay(img, card, cloth)
    overlay_base64 = encode_image(overlay_img)

    # ---------- RESPONSE ----------
    return {
        "pixel_per_inch": round(pixel_per_inch,2),

        "width_in": round(width_in,2),
        "height_in": round(height_in,2),

        "size_width": round(size_width,2),
        "size_height": round(size_height,2),

        "card_box": card,
        "cloth_box": cloth,

        "overlay_image": overlay_base64
    }
