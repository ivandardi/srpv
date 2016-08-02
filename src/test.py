import cv2


def nothing(x):
    pass

img = cv2.imread("../Images/008.jpg")
h, w = img.shape[:2]
img = img[600:h, w / 4:3 * w / 4]
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
img = cv2.bilateralFilter(img, 15, 80, 80)

cv2.namedWindow("a")
cv2.createTrackbar("ThreshL", "a", 0, 350, nothing)
cv2.createTrackbar("ThreshH", "a", 0, 350, nothing)

while True:
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    cimg = img.copy()
    l = cv2.getTrackbarPos("ThreshL", "a")
    h = cv2.getTrackbarPos("ThreshH", "a")
    cimg = cv2.Canny(cimg, l, h)
    cv2.imshow('image', cimg)

cv2.destroyAllWindows()
