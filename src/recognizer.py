import json

from src.pipeline import Package
from src.pipeline.pipeline import Pipeline2
from .util import log_function


class Recognizer:
    @log_function
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        self.pipeline = Pipeline2()

    @log_function
    def recognize(self, img):
        package = Package(img, self.config)
        self.pipeline.apply(package)
        return package
