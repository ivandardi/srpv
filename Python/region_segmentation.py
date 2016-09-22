import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import io
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float
from skimage.morphology import watershed, binary_dilation
from skimage.filters import sobel, threshold_otsu, threshold_adaptive
from skimage.exposure import adjust_sigmoid, adjust_gamma, equalize_adapthist
from skimage.measure import regionprops

from srpv_base import srpv

'''
Constants
'''
# Ratios of plate
RATIO_AVERAGE_PLATE_1 = 2.934
RATIO_STDEV_PLATE_1 = 0.171
RP1M = (RATIO_AVERAGE_PLATE_1 - RATIO_STDEV_PLATE_1)  # * 0.5
RP1P = (RATIO_AVERAGE_PLATE_1 + RATIO_STDEV_PLATE_1)  # * 1.5

RATIO_AVERAGE_PLATE_2 = 2
RATIO_STDEV_PLATE_2 = 0.161
RP2M = (RATIO_AVERAGE_PLATE_2 - RATIO_STDEV_PLATE_2)  # * 0.5
RP2P = (RATIO_AVERAGE_PLATE_2 + RATIO_STDEV_PLATE_2)  # * 1.5

RATIO_AVERAGE_PLATE_3 = 2.511
RATIO_STDEV_PLATE_3 = 0.176
RP3M = (RATIO_AVERAGE_PLATE_3 - RATIO_STDEV_PLATE_3)  # * 0.5
RP3P = (RATIO_AVERAGE_PLATE_3 + RATIO_STDEV_PLATE_3)  # * 1.5

AREA_AVERAGE_PLATE_1 = 304389
AREA_STDEV_PLATE_1 = 93162
AP1M = (AREA_AVERAGE_PLATE_1 - AREA_STDEV_PLATE_1)  # * 0.5
AP1P = (AREA_AVERAGE_PLATE_1 + AREA_STDEV_PLATE_1)  # * 1.5

AREA_AVERAGE_PLATE_2 = 136807
AREA_STDEV_PLATE_2 = 39431
AP2M = (AREA_AVERAGE_PLATE_2 - AREA_STDEV_PLATE_2)  # * 0.5
AP2P = (AREA_AVERAGE_PLATE_2 + AREA_STDEV_PLATE_2)  # * 1.5

AREA_AVERAGE_PLATE_3 = 56398
AREA_STDEV_PLATE_3 = 18667
AP3M = (AREA_AVERAGE_PLATE_3 - AREA_STDEV_PLATE_3)  # * 0.5
AP3P = (AREA_AVERAGE_PLATE_3 + AREA_STDEV_PLATE_3)  # * 1.5


class RegionSegmentation:
    def __init__(self, img_path, dst_path):
        pass

    def run(self):
        print("Running region segmentation")
        img_gray = rgb2gray(equalize_adapthist(self.img.copy(), 5))
        elevation_map = cv2.dilate(sobel(img_gray), np.ones((5, 5), np.uint8), iterations=2)
        io.imsave(self.dst + self.img_name + "_regionsegmentation_elevationmap.jpg", elevation_map)
        markers = np.zeros_like(img_gray)
        otsu = threshold_otsu(img_gray)
        # markers[img_gray < otsu * 0.5] = 1
        markers[img_gray < otsu] = 1
        markers[img_gray >= otsu] = 2
        segmentation = ndi.binary_fill_holes(watershed(elevation_map, markers) - 1)
        binary = segmentation.astype(np.uint8)
        binary[binary > 0] = 255
        io.imsave(self.dst + self.img_name + "_regionsegmentation_binaryfill.jpg", binary)
        labels, _ = ndi.label(segmentation)
        segmented = mark_boundaries(self.img.copy(), labels, color=(0, 1., 1.), mode="thick")
        io.imsave(self.dst + self.img_name + "_regionsegmentation_segmented.jpg", segmented)

    # self.segments,_ = cv2.findContours(labels.astype(dtype=np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    def crop(self):
        pass
