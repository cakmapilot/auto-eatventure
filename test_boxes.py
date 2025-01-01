import subprocess
from PIL import Image
import io
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from adbutils import adb

def find_template(image, template, threshold=0.8):
    if len(image.shape) == 3:image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:image_gray = image

    if len(template.shape) == 3:template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:template_gray = template

    result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    locations = np.where(result >= threshold)
    matches = list(zip(*locations[::-1]))
    
    if len(matches) == 0:
        return None
    coords = np.array(matches)
    dbscan = DBSCAN(eps=10, min_samples=5)
    dbscan.fit(coords)
    centroids = []
    for cluster_label in set(dbscan.labels_):
        if cluster_label != -1:  # Ignore noise points (label -1)
            cluster_coords = coords[dbscan.labels_ == cluster_label]
            centroid = np.mean(cluster_coords, axis=0)
            centroids.append(centroid)

    centroids = [c.astype(int).tolist() for c in centroids]
    return centroids


DEVICE = adb.device()
# Capture screenshot using adb
pipe = subprocess.Popen("adb shell screencap -p",
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, shell=True)
image_bytes = pipe.stdout.read().replace(b'\r\n', b'\n')

# Convert to PIL Image
pilimg = Image.open(io.BytesIO(image_bytes))
pilimg.load()
pilimg = pilimg.convert("RGB")

# Convert PIL image to numpy array (OpenCV format)
open_cv_image = np.array(pilimg)
open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR

current_cv2_sc = open_cv_image
current_cv2_sc_grayscale = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
current_cv2_sc_bgr2hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)

# Save intermediate results
cv2.imwrite("./testfolder/screenshot_current.png", current_cv2_sc)
cv2.imwrite("./testfolder/screenshot_current_grayscale.png", current_cv2_sc_grayscale)
cv2.imwrite("./testfolder/screenshot_current_bgr2hsv.png", current_cv2_sc_bgr2hsv)

# List of template paths
template_paths = [
    "./matching_screenshots/box.png",
    "./matching_screenshots/box1.png",
    "./matching_screenshots/box2.png"
]

# Process each template
for template_path in template_paths:
    print(f"\nProcessing template: {template_path}")
    
    # Load and process template
    templateimg = cv2.imread(template_path)
    if templateimg is None:
        print(f"Failed to load template: {template_path}")
        continue
        
    templateimg_grayscale = cv2.cvtColor(templateimg, cv2.COLOR_BGR2GRAY)
    template_filename = template_path.split('/')[-1].split('.')[0]
    cv2.imwrite(f"./testfolder/{template_filename}_grayscale.png", templateimg_grayscale)

    # Try different thresholds
    threshold = 0.8
    print(f"\nTrying threshold: {threshold}")
    match_locations = find_template(current_cv2_sc, templateimg, threshold=threshold)
    if match_locations and len(match_locations) > 0:
        for match_location in match_locations:
            match_location_tmp = (match_location[0]+60, match_location[1]+60)
            print(f"Match found at {match_location_tmp} with threshold {threshold}")
            DEVICE.click(match_location_tmp[0], match_location_tmp[1])
            h, w = current_cv2_sc.shape[:2]
            top_left = match_location_tmp
        bottom_right = (top_left[0] + w, top_left[1] + h)
        result_image = current_cv2_sc.copy()
        cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.imwrite(f"./testfolder/{template_filename}_result_threshold_{threshold}.png", result_image)





