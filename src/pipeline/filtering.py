import cv2
from util import Rect, draw_rectangles_on_image

from . import Package


class BilateralFilter:
    def __init__(self, diameter, sigma_color, sigma_space):
        """"
        Parameters
        ----------
        diameter
            Diameter of each pixel neighborhood that is used during filtering.
            If it is non-positive, it is computed from sigma_space.
        sigma_color
            Filter sigma in the color space. A larger value of the parameter
            means that farther colors within the pixel neighborhood (see sigma_space)
            will be mixed together, resulting in larger areas of semi-equal color.
        sigma_space
            Filter sigma in the coordinate space. A larger value of the parameter
            means that farther pixels will influence each other as long as their
            colors are close enough (see sigma_color ). When diameter>0, it specifies the
            neighborhood size regardless of sigma_space. Otherwise, diameter is proportional
            to sigma_space.
        """
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, package: Package):
        package.images['BilateralFilter'] = cv2.bilateralFilter(
            package.current_image,
            self.diameter,
            self.sigma_color,
            self.sigma_space)


class FilterCharacterRatio:
    def __init__(self, min_precision, max_precision):

        """
        Filters contours, extracting only the ones that
        look like characters in dimension and proportion
        and returns the bounding rectangles of those
        characters.
        """
        self.min_precision = min_precision
        self.max_precision = max_precision

    def __call__(self, package: Package):
        ideal_character_width = 50
        ideal_character_height = 79
        ideal_character_ratio = ideal_character_width / ideal_character_height

        character_rects = []
        for cnt in package.latest_pipeline_data[0]:
            rect = Rect(cv2.boundingRect(cnt))
            actual_char_ratio = rect.w / rect.h
            char_precision = actual_char_ratio / ideal_character_ratio
            if self.min_precision <= char_precision <= self.max_precision:
                character_rects.append(rect)

        package.pipeline_data['CharacterFilter'] = character_rects
        package.images['CharacterFilter'] = draw_rectangles_on_image(package.threshold_image, character_rects)


class FilterCharacterSize:
    def __init__(self, min_area, max_area):
        self.min_area = min_area
        self.max_area = max_area

    def __call__(self, package: Package):
        def area_predicate(r):
            return self.min_area < r.area() < self.max_area

        package.pipeline_data['SizeFilter'] = list(filter(area_predicate, package.latest_pipeline_data))
        package.images['SizeFilter'] = draw_rectangles_on_image(package.threshold_image, package.latest_pipeline_data)
