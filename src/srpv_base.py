import cv2


# noinspection SpellCheckingInspection
class srpv:
    def __init__(self, img):
        self.image = img

        h, w = img.get_image_dimensions()
        self.img_cropped = self.image.get_color_image_copy()[600:h, int(w * 0.25):int(w * 0.75)]
        self.img_cropped = cv2.resize(self.img_cropped, (0, 0), fx=0.5, fy=0.5)
        #  816 x 924

        self.img_gray = cv2.cvtColor(self.img_cropped, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("{0}/{1}_gray.jpg".format(self.image.path_save, self.image.name), self.img_gray)

    def run(self):
        raise NotImplementedError('users must define run to use this base class')

    def crop(self):
        raise NotImplementedError('users must define crop to use this base class')
