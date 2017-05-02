import cv2

from . import Pipeline, Package


class CLAHE(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.clahe = cv2.createCLAHE(*args, **kwargs)

    def apply(self, package: Package):
        package.images['CLAHE'] = self.clahe.apply(package.current_image)
