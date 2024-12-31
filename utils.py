import cv2

def imshow(cv2im, makeitsmall=True):
    cv2.namedWindow("imim", cv2.WINDOW_NORMAL)
    if makeitsmall:
        im = cv2.resize(cv2im, (504, 1092))
    else:
        im = cv2im
    cv2.imshow("imim", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()