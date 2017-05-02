from abc import ABC
from collections import OrderedDict

import numpy as np

from src.util import log_function


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
    config
        Config values for the pipeline
    """

    def __init__(self, original_image: np.ndarray, config: str):
        self.images = OrderedDict()
        self.images['original'] = original_image
        self.pipeline_data = OrderedDict()
        self.config = config

    @property
    def current_image(self):
        return next(reversed(self.images.values()))

    @property
    def original_image(self):
        return self.images['Thumbnail'].copy()

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

    def __init__(self, *functions):
        self.pipeline = functions

    @log_function
    def apply(self, package: Package):
        for pipeline in self.pipeline:
            pipeline.apply(package)
