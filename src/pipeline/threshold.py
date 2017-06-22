import logging

import numpy as np
from skimage.filters.thresholding import _mean_std as mean_std

from . import Package

log = logging.getLogger(__name__)


class WolfJolionThreshold:
    def __init__(self, window_size=15, k=0.2):
        self.window_size = window_size
        self.k = k

    def __call__(self, package: Package):
        image = package.current_image
        m, s = mean_std(image, self.window_size)
        mask = m + self.k * ((s / np.max(s)) - 1) * (m - np.min(image))
        thresh = (image > mask) * 255
        package.images['Threshold'] = thresh.astype(dtype=np.uint8)
