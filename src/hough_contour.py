import cv2
import numpy as np

from srpv_base import srpv


class HoughContour(srpv):
    def __init__(self, img_path, dst_path):
        srpv.__init__(self, img_path, dst_path)

    def run(self):
        # Extract contours from image
        cont, _ = cv2.findContours(self.img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        

        # Detect paralel lines with Hough

    def crop(self):
        raise NotImplementedError('users must define crop to use this base class')

