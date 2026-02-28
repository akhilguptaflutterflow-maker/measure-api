from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np

app = FastAPI()

# ================= INPUT =================
class ImageData(BaseModel):
    image: str


# ================= BASE64 =================
def decode_image(base64_str):
    header_removed = base64_str.split(",")[-1]
    img_bytes = base64.b64decode(header_removed)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')


# ================= CARD DETECTION (IMPROVED) =================
def detect_card(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,50,150)

    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = img.shape[:2]
    target_y = H * 0.65   # card usually lower half

    best = None
    best_score = 999

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        # basic size filter
        if w < 120 or h < 70:
            continue

        ratio = w / h
        ratio_diff = abs(ratio - 1.586)

        # distance from lower center
        cy = y + h/2
        dist = abs(cy - target_y) / H

        score = ratio_diff + dist

        if score < best_score:
            best_score = score
            best = (x,y,w,h)

    return best


# ================= CLOTH SEGMENT =================
def segment_cloth(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)[1]

    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, -1)

    x,y,w,h = cv2.boundingRect(largest)

    return mask, (x,y,w,h)


# ================= WIDTH AT ROW =================
def width_at_y(mask, y):
    row = mask[y]
    xs = np.where(row > 0)[0]
    if len(xs) < 2:
        return 0
    return xs[-1] - xs[0]


# ================= MAIN API =================
@app.post("/measure")
def measure(data: ImageData):

    img = decode_image(data.image)

    # -------- CARD --------
    card = detect_card(img)
    if card is None:
        return {"error": "Card not detected"}

    x,y,w,h = card
    pixel_per_inch = w / 3.37

    # -------- CLOTH --------
    mask, (cx,cy,cw,ch) = segment_cloth(img)

    top = cy
    bottom = cy + ch

    # better vertical sampling
    waist_y  = int(top + ch*0.02)
    hip_y    = int(top + ch*0.30)
    bottom_y = int(top + ch*0.95)

    waist_px  = width_at_y(mask, waist_y)
    hip_px    = width_at_y(mask, hip_y)
    bottom_px = width_at_y(mask, bottom_y)
    length_px = ch

    # -------- CONVERT (IMPORTANT ×2 for flat cloth) --------
    waist_in  = (waist_px / pixel_per_inch) * 2
    hip_in    = (hip_px / pixel_per_inch) * 2
    bottom_in = (bottom_px / pixel_per_inch) * 2
    length_in = length_px / pixel_per_inch

    # -------- DRAW OVERLAY --------
    overlay = img.copy()

    # card box
    cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.putText(overlay,"CARD",(x,y-10),0,0.8,(0,255,0),2)

    # waist line
    cv2.line(overlay,(0,waist_y),(overlay.shape[1],waist_y),(255,0,0),2)
    cv2.putText(overlay,f"WAIST {waist_in:.1f} in",(10,waist_y-10),0,0.7,(255,0,0),2)

    # hip
    cv2.line(overlay,(0,hip_y),(overlay.shape[1],hip_y),(0,255,255),2)
    cv2.putText(overlay,f"HIP {hip_in:.1f} in",(10,hip_y-10),0,0.7,(0,255,255),2)

    # bottom
    cv2.line(overlay,(0,bottom_y),(overlay.shape[1],bottom_y),(0,0,255),2)
    cv2.putText(overlay,f"BOTTOM {bottom_in:.1f} in",(10,bottom_y-10),0,0.7,(0,0,255),2)

    # length line
    cv2.line(overlay,(cx,cy),(cx,cy+ch),(255,255,0),3)
    cv2.putText(overlay,f"LENGTH {length_in:.1f} in",(cx+10,cy+ch//2),0,0.7,(255,255,0),2)

    overlay_base64 = encode_image(overlay)

    # -------- RESPONSE --------
    return {
        "pixel_per_inch": round(pixel_per_inch,2),
        "waist_in": round(waist_in,2),
        "hip_in": round(hip_in,2),
        "bottom_in": round(bottom_in,2),
        "length_in": round(length_in,2),
        "overlay_image": overlay_base64
    }
