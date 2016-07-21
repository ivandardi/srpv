import os
from datetime import datetime as dt

from adaptive_segmentation import AdaptiveSegmentation
from edge_segmentation import EdgeSegmentation
from region_segmentation import RegionSegmentation

'''Diretorios das imagens e das analises'''
IMG_DIR = "../Images/"
ANA_DIR = "../Analises/"

'''Path of analisis using the date'''
date = dt.now()
path = ANA_DIR + str(date.year) + "." + str(date.month).zfill(2) + "." + str(date.day).zfill(2) + "-" + str(
    date.hour).zfill(2) + "." + str(date.minute).zfill(2) + "." + str(date.second).zfill(2) + "/"

def main():
    os.mkdir(path)

    print("Analysis " + path)
    for i in sorted(os.listdir(IMG_DIR)):
        print("Analising image: " + i)
        RegionSegmentation(IMG_DIR + i, path).run()
        # an = plate_detection.AdaptiveSegmentation(IMG_DIR + i, path)
        # an = plate_detection.EdgeSegmentation(IMG_DIR + i, path)


if __name__ == "__main__":
    main()
