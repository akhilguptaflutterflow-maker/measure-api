from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np

app = FastAPI()

class ImageData(BaseModel):
    image: str
    garment_type: str

def decode_image(base64_str):
    header_removed = base64_str.split(",")[-1]
    img_bytes = base64.b64decode(header_removed)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

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

        if diff < best_ratio_diff and w>50 and h>30:
            best = (x,y,w,h)
            best_ratio_diff = diff

    return best

def find_cloth(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,50,150)

    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(largest)
    return (x,y,w,h)

@app.post("/measure")
def measure(data: ImageData):
    img = decode_image(data.image)

    card = find_card(img)
    cloth = find_cloth(img)

    if card is None:
        return {"error": "Card not detected"}

    x,y,w,h = card
    pixel_per_inch = w / 3.37

    cx,cy,cw,ch = cloth

    width_in = cw / pixel_per_inch
    height_in = ch / pixel_per_inch

    return {
        "pixel_per_inch": round(pixel_per_inch,2),
        "width_in": round(width_in,2),
        "height_in": round(height_in,2),
        "size_width": round(width_in*2,2),
        "size_height": round(height_in*2,2)
    }
