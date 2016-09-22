import os
from datetime import datetime as dt
import cv2
from image import image
from hough_contour import HoughContour


def main():
    IMG_DIR = "../Images"
    ANA_DIR = "../Analises"

    '''Path of analisis using the date'''
    date = dt.now()
    dst_path = "{}/{:0>2}.{:0>2}.{:0>2}-{:0>2}.{:0>2}.{:0>2}".format(ANA_DIR,
                                                                     date.year,
                                                                     date.month,
                                                                     date.day,
                                                                     date.hour,
                                                                     date.minute,
                                                                     date.second)

    os.mkdir(dst_path)
    os.mkdir("{}/crops".format(dst_path))

    print("Analysis " + dst_path)
    for i in sorted(os.listdir(IMG_DIR)):
        print("Analyzing image: " + i)
        img = image(cv2.imread("{}/{}".format(IMG_DIR, i)), i[-7:-4], dst_path, "{}/crops".format(dst_path))
        HoughContour(img).run()

if __name__ == "__main__":
    main()
