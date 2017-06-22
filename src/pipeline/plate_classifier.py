import logging
import string

import cv2
import numpy as np

from . import Pipeline, Package

log = logging.getLogger(__name__)


class DetectPlate(Pipeline):
    def __init__(self):
        super().__init__()
        self.ocr = cv2.text.OCRTesseract_create(
            language='por',
            char_whitelist=string.ascii_uppercase + string.digits,
            psmode=7,
        )


    def apply(self, package: Package):
        images = package.latest_pipeline_data
        package.pipeline_data['DetectPlate'] = [self.ocr.run(img, np.ones_like(img), 0) for img in images]
        log.info('OCR')
        log.info(package.pipeline_data['DetectPlate'])
