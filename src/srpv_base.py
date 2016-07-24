import cv2


class srpv():

    def __init__(self, img_path, dst_path):
        self.path = img_path
        self.dst = dst_path
        self.img_name = img_path[-7:-4]
        self.img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        h, w = self.img.shape
        self.img = self.img[600:h, w / 4:3 * w / 4]
        cv2.imwrite("{0}{1}_preprocessing1.jpg".format(self.dst, self.img_name), self.img)

        # Preprocessing
        #self.img = cv2.normalize(cv2.calcHist([self.img], [0], None, [256], [0,256]))
        self.img = cv2.GaussianBlur(self.img, (9,9), 0)
        cv2.imwrite("{0}{1}_preprocessing2.jpg".format(self.dst, self.img_name), self.img)
        self.img = cv2.equalizeHist(self.img)
        cv2.imwrite("{0}{1}_preprocessing3.jpg".format(self.dst, self.img_name), self.img)
        self.img = cv2.Sobel(self.img, -1, 0, 1, ksize=5)
        cv2.imwrite("{0}{1}_preprocessing4.jpg".format(self.dst, self.img_name), self.img)
        self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)
        cv2.imwrite("{0}{1}_preprocessing5.jpg".format(self.dst, self.img_name), self.img)


    def run(self):
        raise NotImplementedError('users must define run to use this base class')


    def crop(self):
        raise NotImplementedError('users must define crop to use this base class')
