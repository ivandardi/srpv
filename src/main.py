import os
from datetime import datetime as dt

from adaptive_segmentation import AdaptiveSegmentation
from edge_segmentation import EdgeSegmentation
from region_segmentation import RegionSegmentation
from hough_contour import HoughContour

IMG_DIR = "../Images/"
ANA_DIR = "../Analises/"

'''Path of analisis using the date'''
date = dt.now()
dst_path = ANA_DIR + str(date.year) + "." + str(date.month).zfill(2) + "." + str(date.day).zfill(2) + "-" + str(
    date.hour).zfill(2) + "." + str(date.minute).zfill(2) + "." + str(date.second).zfill(2) + "/"


def main():
    os.mkdir(dst_path)

    print("Analysis " + dst_path)
    for i in sorted(os.listdir(IMG_DIR)):
        print("Analising image: " + i)
        HoughContour(IMG_DIR + i, dst_path).run()

if __name__ == "__main__":
    main()
