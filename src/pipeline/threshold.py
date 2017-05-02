import cv2
import matplotlib.pyplot as plt

from . import Pipeline, Package


# TODO add Wolfjolion thresholding

class OtsuThreshold(Pipeline):
    def __init__(self):
        super().__init__()

    def apply(self, package: Package):
        fig, ax = plt.subplots()
        ax.hist(package.current_image.ravel(), 256)
        fig.show()
        otsu_value, package.images['Threshold'] = cv2.threshold(package.current_image, 0, 255,
                                                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)


class AdaptiveThreshold(Pipeline):
    def __init__(self, thresholdType, blockSize, C):
        """
        thresholdType	Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV, see cv::ThresholdTypes.
        blockSize	Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
        C	Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
        """
        super().__init__()
        self.thresholdType = thresholdType
        self.blockSize = blockSize
        self.C = C

    def apply(self, package: Package):
        fig, ax = plt.subplots()
        ax.hist(package.current_image.ravel(), 256)
        fig.show()
        package.images['Threshold'] = cv2.adaptiveThreshold(package.current_image, 255,
                                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                            self.thresholdType, self.blockSize, self.C)
