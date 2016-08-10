from skimage import io
from skimage.exposure import equalize_adapthist
from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float
from srpv_base import srpv
from constants import *


class AdaptiveSegmentation(srpv):
    def __init__(self, img_path, dst_path):
        self.path = img_path
        self.dst = dst_path
        self.img_name = img_path[-7:-4]
        self.img = io.imread(self.path)
        h, w = self.img.shape[:2]
        self.img = img_as_float(self.img[600:h, w / 4:3 * w / 4])

    def run(self, comp=10.0):
        print("Running slic equalized")
        equalized = equalize_adapthist(self.img, 5)
        io.imsave(self.dst + self.img_name + "_adaptivesegmentation_equalized.jpg", equalized)
        self.segments = slic(equalized, n_segments=10, compactness=comp, sigma=5, max_iter=20)
        segmented = mark_boundaries(self.img, self.segments, color=(0, 1., 1.), mode="thick")
        io.imsave(self.dst + self.img_name + "_adaptivesegmentation_segmented.jpg", segmented)

    def crop(self):
        pass
