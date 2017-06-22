import logging
from abc import ABC
from collections import OrderedDict
from typing import Callable, Iterable

import cv2
import numpy as np

log = logging.getLogger(__name__)


class Package:
    """
    Object that is passed around in the pipeline.

    Attributes
    ----------
    images : OrderedDict
        Stack-like container of images. Used to save the pipeline of images
    pipeline_data : OrderedDict
        Stack-like container of pipeline data. Used to transfer results from
        one pipeline step to the other

    Properties
    ----------
    current_image : np.ndarray
        Current image being processed in the pipeline
    original_image : np.ndarray
        Copy of original RGB image for drawing on
    latest_pipeline_data : OrderedDict[str, Package]
        Latest pipeline data, packaged in a package
    """

    def __init__(self, original_image: np.ndarray):
        self.images = OrderedDict()
        self.images['Original'] = original_image
        self.pipeline_data = OrderedDict()

    @property
    def current_image(self):
        return next(reversed(self.images.values()))

    @property
    def original_image(self):
        return self.images.get('Thumbnail', self.images['Original']).copy()

    @property
    def threshold_image(self):
        return cv2.cvtColor(self.images.get('Threshold', self.images['Original']), cv2.COLOR_GRAY2BGR)

    @property
    def latest_pipeline_data(self):
        return next(reversed(self.pipeline_data.values()))


class Pipeline(ABC):
    """
    This class implements the Composite design pattern.
    It allows for arbitrary chaining of processing
    functions and any grouping of them.

    The objective is to take the original image,
    put it through the pipeline and hopefully
    get the license place from the image
    """

    def __init__(self, name: str, functions: Iterable[Callable]):
        self.name = name
        self.pipeline = functions

    def __call__(self, package: Package):
        for fn in self.pipeline:
            fn(package)
