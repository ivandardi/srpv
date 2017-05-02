import logging
from functools import wraps

import cv2

log = logging.getLogger(__name__)


def enter_only(func):
    @wraps(func)
    def wrapper(*func_args, **func_kwargs):
        name = func.__qualname__
        log.info(f'ENTER: {name}')
        return func(*func_args, **func_kwargs)

    return wrapper


def enter_exit(func):
    @wraps(func)
    def wrapper(*func_args, **func_kwargs):
        name = func.__module__ + func.__qualname__
        log.info(f'ENTER: {name}')
        ret = func(*func_args, **func_kwargs)
        log.info(f' EXIT: {name}')
        return ret

    return wrapper


def dont_log_names(func):
    @wraps(func)
    def wrapper(*func_args, **func_kwargs):
        return func(*func_args, **func_kwargs)

    return wrapper


log_function = enter_exit


def draw_rectangles_on_image(image, rectangles):
    for r in rectangles:
        r.draw(image)
    return image


class Rect:
    def __init__(self, *args):
        self.x, self.y, self.w, self.h = args[0] if len(args) == 1 else args

    def __repr__(self):
        return f'Rect({self.__dict__})'

    def area(self):
        return self.w * self.h

    def center(self):
        return self.x + self.w / 2, self.y + self.h / 2

    @property
    def top_left(self):
        return self.x, self.y

    @property
    def bottom_right(self):
        return self.x + self.w, self.y + self.h

    @property
    def top_right(self):
        return self.x + self.w, self.y

    @property
    def bottom_left(self):
        return self.x, self.y + self.h

    def draw(self, img, color=(0, 0, 0xFF), thickness=1):
        cv2.rectangle(img, self.top_left, self.bottom_right, color, thickness)


class Oval:
    def __init__(self, *args):
        self.x, self.y, self.major_axis, self.minor_axis, self.angle = args[0] if len(args) == 1 else args
