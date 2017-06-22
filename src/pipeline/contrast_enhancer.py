import cv2

from .definitions import Package


def equalize_histogram(package: Package, *args, **kwargs):
    package.images['CLAHE'] = cv2.createCLAHE(*args, **kwargs).apply(package.current_image)
