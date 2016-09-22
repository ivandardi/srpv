import cv2
import numpy as np

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
black = (0, 0, 0)
white = (255, 255, 255)


def nothing(x):
    pass

img = cv2.imread("../Images/008.jpg")
h, w = img.shape[:2]
img = img[600:h, int(w * 0.25):int(w * 0.75)]
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
img = cv2.bilateralFilter(img, 15, 80, 80)
img = cv2.Canny(img, 80, 240)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=10)
img_contours, contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cv2.namedWindow("a")
cv2.createTrackbar("Epsilon", "a", 0, 10000, nothing)

while True:
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    cimg = np.zeros_like(img)
    e = cv2.getTrackbarPos("Epsilon", "a")

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, e * 0.01, True)
        cv2.fillPoly(cimg, approx, white)

    cv2.imshow('image', cimg)

cv2.destroyAllWindows()
