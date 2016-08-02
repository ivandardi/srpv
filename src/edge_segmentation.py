import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import canny
from srpv_base import srpv
from constants import *


class EdgeSegmentation(srpv):
    def __init__(self, img_path, dst_path):
        self.path = img_path
        self.dst = dst_path
        self.img_name = img_path[-7:-4]
        self.img = io.imread(self.path)
        h, w = self.img.shape[:2]
        self.img = self.img[600:h, w / 4:3 * w / 4]

    def run(self):
        print("Running edge segmentation")
        img_gray = rgb2gray(self.img)
        edges = canny(img_gray)
        io.imsave(self.dst + self.img_name + "_edgesegmentation_canny.jpg", edges)
        filled = ndi.binary_fill_holes(edges)
        io.imsave(self.dst + self.img_name + "_edgesegmentation_filled.jpg", filled)
        labels, _ = ndi.label(filled)
        sizes = np.bincount(labels.ravel())
        mask_sizes = sizes > 20
        mask_sizes[0] = 0
        cleaned = mask_sizes[labels]
        io.imsave(self.dst + self.img_name + "_edgesegmentation_labels.jpg", cleaned)

    # self.segments,_ = cv2.findContours(labels.astype(dtype=np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    def crop(self):
        i = 0
        rect_img = self.img.copy()
        for cnt in self.segments:
            [x, y, w, h] = cv2.boundingRect(cnt)
            plate_ratio = float(w) / h
            area = w * h
            if AP1M < area < AP1P:
                if RP1M < plate_ratio < RP1P:
                    cv2.imwrite(self.dst + self.img_name + "_crop_" + str(i) + ".jpg", self.img[y:y + h, x:x + w])
                    cv2.rectangle(rect_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
            elif AP2M < area < AP2P:
                if RP2M < plate_ratio < RP2P:
                    cv2.imwrite(self.dst + self.img_name + "_crop_" + str(i) + ".jpg", self.img[y:y + h, x:x + w])
                    cv2.rectangle(rect_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
            elif AP3M < area < AP3P:
                if RP3M < plate_ratio < RP3P:
                    cv2.imwrite(self.dst + self.img_name + "_crop_" + str(i) + ".jpg", self.img[y:y + h, x:x + w])
                    cv2.rectangle(rect_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
            i += 1
        cv2.imwrite(self.dst + self.img_name + "_rectangles.jpg", self.img[y:y + h, x:x + w])
