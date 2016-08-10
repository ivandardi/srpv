import cv2


class image:
    def __init__(self, img, name, path_save, path_crop):
        self._img = img
        self.name = name
        self.path_save = path_save
        self.path_crop = path_crop

    def get_color_image_copy(self):
        return self._img.copy()

    def get_grayscale_image(self):
        return cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

    def get_image_dimensions(self):
        return self._img.shape[:2]



