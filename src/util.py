import logging

import cv2

log = logging.getLogger(__name__)


def draw_rectangles_on_image(image, rectangles):
    for r in rectangles:
        r.draw(image)
    return image


def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


class ABCRect:
    @property
    def top_left(self):
        return self.left, self.top

    @property
    def bottom_right(self):
        return self.right, self.bottom

    @property
    def top_right(self):
        return self.right, self.top

    @property
    def bottom_left(self):
        return self.left, self.bottom

    def draw(self, img, color=(0, 0, 0xFF), thickness=3):
        cv2.rectangle(img, self.top_left, self.bottom_right, color, thickness)

    def area(self):
        return self.h * self.w

    def center(self):
        return self.x + self.w / 2, self.y + self.h / 2

    def size(self):
        return self.h, self.w


class Rect(ABCRect):
    def __init__(self, *args):
        self.x, self.y, self.w, self.h = args[0] if len(args) == 1 else args
        self.top = self.y
        self.bottom = self.top + self.h
        self.left = self.x
        self.right = self.left + self.w

    def __repr__(self):
        return f'Rect({self.__dict__})'


class Region(ABCRect):
    def __init__(self, cluster):
        self.top = min(rect.top for rect in cluster)
        self.bottom = max(rect.bottom for rect in cluster)
        self.left = min(rect.left for rect in cluster)
        self.right = max(rect.right for rect in cluster)

    def __repr__(self):
        return f'Region({self.__dict__})'


class Oval:
    def __init__(self, *args):
        self.x, self.y, self.major_axis, self.minor_axis, self.angle = args[0] if len(args) == 1 else args
