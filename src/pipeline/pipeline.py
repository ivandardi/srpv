import logging

import cv2

from src.pipeline.misc import PlateFinder
from src.pipeline.plate_classifier import DetectPlate
from . import misc, contrast_enhancer, filter, threshold, Pipeline

log = logging.getLogger(__name__)


class Pipeline1(Pipeline):
    def __init__(self):
        super().__init__(
            misc.GrayscaleThumbnail(),
            contrast_enhancer.CLAHE(),
            filter.BilateralFilter(5, 40, 40),
            threshold.AdaptiveThreshold(cv2.THRESH_BINARY, 3, 2),
            PlateFinder(),
            DetectPlate(),
        )


class Pipeline2(Pipeline):
    def __init__(self):
        super().__init__(
            misc.GrayscaleThumbnail(),
            contrast_enhancer.CLAHE(),
            filter.BilateralFilter(5, 40, 40),
            threshold.OtsuThreshold(),
            PlateFinder(),
            DetectPlate(),
        )
