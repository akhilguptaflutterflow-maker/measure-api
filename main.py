from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np

app = FastAPI()

# ---------------- INPUT ----------------
class ImageData(BaseModel):
    image: str
    pixel_per_inch: float


# ---------------- BASE64 ----------------
def decode_image(base64_str):
    header_removed = base64_str.split(",")[-1]
    img_bytes = base64.b64decode(header_removed)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')


# ---------------- SEGMENT CLOTH (EDGE BASED - STRONG) ----------------
def segment_cloth(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blur, 40, 120)

    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, -1)

    x,y,w,h = cv2.boundingRect(largest)

    return mask,(x,y,w,h)


# ---------------- WIDTH AT ROW (GAP FILL) ----------------
def width_at_y(mask, y):
    row = mask[y]

    # fill gaps in waistband area
    kernel = np.ones((1,25), np.uint8)
    row = cv2.dilate(row.reshape(1,-1), kernel, iterations=1)[0]

    xs = np.where(row > 0)[0]
    if len(xs) < 2:
        return 0

    return xs[-1] - xs[0]


# ---------------- MAIN API ----------------
@app.post("/measure")
def measure(data: ImageData):

    if data.pixel_per_inch <= 0:
        return {"error": "Invalid pixel_per_inch"}

    img = decode_image(data.image)

    mask, box = segment_cloth(img)

    if mask is None:
        return {"error": "Cloth not detected"}

    cx, cy, cw, ch = box
    top = cy

    # 👇 UPDATED ROW POSITIONS (FIXED)
    waist_y  = int(top + ch*0.08)
    hip_y    = int(top + ch*0.30)
    bottom_y = int(top + ch*0.95)

    waist_px  = width_at_y(mask, waist_y)
    hip_px    = width_at_y(mask, hip_y)
    bottom_px = width_at_y(mask, bottom_y)
    length_px = ch

    ppi = data.pixel_per_inch

    # 👇 flat garment multiply ×2
    waist_in  = (waist_px / ppi) * 2
    hip_in    = (hip_px / ppi) * 2
    bottom_in = (bottom_px / ppi) * 2
    length_in = length_px / ppi

    # ---------------- DRAW OVERLAY ----------------
    overlay = img.copy()

    cv2.line(overlay,(0,waist_y),(overlay.shape[1],waist_y),(255,0,0),2)
    cv2.putText(overlay,f"WAIST {waist_in:.1f} in",(10,waist_y-10),0,0.7,(255,0,0),2)

    cv2.line(overlay,(0,hip_y),(overlay.shape[1],hip_y),(0,255,255),2)
    cv2.putText(overlay,f"HIP {hip_in:.1f} in",(10,hip_y-10),0,0.7,(0,255,255),2)

    cv2.line(overlay,(0,bottom_y),(overlay.shape[1],bottom_y),(0,0,255),2)
    cv2.putText(overlay,f"BOTTOM {bottom_in:.1f} in",(10,bottom_y-10),0,0.7,(0,0,255),2)

    cv2.line(overlay,(cx,cy),(cx,cy+ch),(255,255,0),3)
    cv2.putText(overlay,f"LENGTH {length_in:.1f} in",(cx+10,cy+ch//2),0,0.7,(255,255,0),2)

    overlay_base64 = encode_image(overlay)

    # ---------------- RESPONSE ----------------
    return {
        "pixel_per_inch": round(ppi,2),
        "waist_in": round(waist_in,2),
        "hip_in": round(hip_in,2),
        "bottom_in": round(bottom_in,2),
        "length_in": round(length_in,2),
        "overlay_image": overlay_base64
    }
