import cv2

from src.util import Rect, Oval, draw_rectangles_on_image
from . import Pipeline, Package


class BilateralFilter(Pipeline):
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
        super().__init__()
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply(self, package: Package):
        package.images['BilateralFilter'] = cv2.bilateralFilter(package.current_image, self.diameter,
                                                                self.sigma_color, self.sigma_space)


class CharacterFilter:
    """
    Filters contours, extracting only the ones that
    look like characters in dimension and proportion
    and returns the bounding rectangles of those
    characters.
    """

    def __init__(self):
        super().__init__()
        self.ideal_character_width = 50
        self.ideal_character_height = 79
        self.ideal_character_ratio = self.ideal_character_width / self.ideal_character_height


class CharacterRatioFilter(CharacterFilter, Pipeline):
    """
    Should only be called via CharacterContours
    """

    def apply(self, package: Package):
        character_rects = []
        for cnt in package.latest_pipeline_data[0]:
            rect = Rect(cv2.boundingRect(cnt))
            actual_char_ratio = rect.w / rect.h
            char_precision = actual_char_ratio / self.ideal_character_ratio
            if package.config['min_precision'] <= char_precision <= package.config['max_precision']:
                character_rects.append(rect)

        package.pipeline_data['CharacterFilter'] = character_rects
        package.images['CharacterFilter'] = draw_rectangles_on_image(package.original_image, character_rects)


class CharacterOvalFilter(CharacterFilter, Pipeline):
    """
    Should only be called via CharacterContours
    """

    def apply(self, package: Package):
        """
        Gets the ellipse that the contour fits in.
        It checks to see if it's standing upright, because we
        don't want rectangles that are askew.
        If it's an acceptable ellipse, we calculate the
        bounding rectangle of it.
         
        """
        package.character_rects = []
        for cnt in package.contours.contours:
            oval = Oval(cv2.fitEllipse(cnt))
            actual_char_ratio = oval.minor_axis / oval.major_axis
            char_precision = actual_char_ratio / self.ideal_character_ratio
            if package.config['min_precision'] <= char_precision <= package.config['max_precision']:
                if 80 <= oval.angle <= 110:
                    # TODO fine tune this range
                    package.character_rects.append(Rect(cv2.boundingRect(cnt)))


class SizeFilter(Pipeline):
    def __init__(self):
        super().__init__()

    def apply(self, package: Package):
        def area_predicate(r):
            return package.config['min_area'] < r.area() < package.config['max_area']

        package.pipeline_data['SizeFilter'] = list(filter(area_predicate, package.latest_pipeline_data))
        package.images['SizeFilter'] = draw_rectangles_on_image(package.original_image, package.latest_pipeline_data)
