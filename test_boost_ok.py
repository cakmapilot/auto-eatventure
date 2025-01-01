import glob
import os
import subprocess
import io
from PIL import Image
import numpy as np
import cv2
from adbutils import adb

DEVICE = adb.device()

def check_boost_text(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return False
        
    roi = img[3060:3100, 640:800]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_pixel_count = np.sum(thresh == 255)
    return white_pixel_count > 0


check_images = glob.glob("./testfolder/*.png")

for image_path in check_images:
    has_boost = check_boost_text(image_path)
    print(f"{os.path.basename(image_path)}: {'Has boost text' if has_boost else 'No boost text'}")
