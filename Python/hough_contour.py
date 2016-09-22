import cv2
import numpy as np
from srpv_base import srpv
from constants import *


def draw_rect(img, cnt, color=white, thickness=2):
    rect = cv2.minAreaRect(cnt)
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(img, [box], 0, color, thickness)


class HoughContour(srpv):
    def __init__(self, img):
        srpv.__init__(self, img)

    def run(self):
        img_contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(self.img_gray)
        cv2.imwrite("{0}/{1}_preprocessing1.jpg".format(self.image.path_save, self.image.name), img_contrast)

        # Preprocessing
        img_blur = cv2.bilateralFilter(img_contrast, 15, 80, 80)
        cv2.imwrite("{0}/{1}_preprocessing2.jpg".format(self.image.path_save, self.image.name), img_blur)

        img_canny = cv2.Canny(img_blur, 80, 240)
        img_canny = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=10)
        cv2.imwrite("{0}/{1}_preprocessing3.jpg".format(self.image.path_save, self.image.name), img_canny)

        img_contours, contours, hierarchy = cv2.findContours(img_canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        img_filled = np.zeros_like(self.img_gray)
        cv2.drawContours(img_filled, contours, -1, white, 2)
        cv2.imwrite("{0}/{1}_preprocessing4.jpg".format(self.image.path_save, self.image.name), img_filled)

        img_filled = np.zeros_like(self.img_gray)
        for cnt in contours:
            # shapes = cv2.approxPolyDP(cnt, 1.0, True)
            shape = cv2.convexHull(cnt, returnPoints=True)
            cv2.fillPoly(img_filled, shape, white)
        cv2.imwrite("{0}/{1}_preprocessing5.jpg".format(self.image.path_save, self.image.name), img_filled)

        lines = cv2.HoughLines(img_filled, 1, np.pi / 180, 75)
        try:
            for i in lines:
                rho = i[0][0]
                theta = i[0][1]
                angle = theta * (180 / np.pi)
                draw = False
                if angle < 10 or angle > 350 or (170 < angle < 190):
                    #  good vertical
                    draw = True
                if (80 < angle < 110) or (260 < angle < 280):
                    #  good horizontal
                    draw = True
                if draw:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * a)
                    cv2.line(img_canny, (x1, y1), (x2, y2), white)
        except Exception as e:
            print(e)

        cv2.imwrite("{0}/{1}_preprocessing6.jpg".format(self.image.path_save, self.image.name), img_contours)

        # img_rect = self.img_orig.copy()
        # img_cont = np.zeros_like(self.img_orig)
        #
        # cv2.imwrite("{0}{1}_processing_contour.jpg".format(self.dst, self.img_name, i), img_cont)
        # cv2.imwrite("{0}{1}_processing_rectangle.jpg".format(self.dst, self.img_name, i), img_rect)

    def crop(self):
        pass
