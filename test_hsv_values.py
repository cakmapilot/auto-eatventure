import cv2

# Add this before the box_mask_image function
def get_hsv_values(image_path):
    img = cv2.imread(image_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            pixel = hsv_img[y, x]
            print(f"HSV values at ({x},{y}): {pixel}")
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)
    
    while True:
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

# Use it like this:
get_hsv_values("./matching_screenshots/box.png")