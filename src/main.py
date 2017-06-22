import json
import logging
import os
from time import gmtime, strftime

import cv2
import numpy as np
from imutils import build_montages
from pipeline import *

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
np.set_printoptions(threshold=np.nan, linewidth=np.nan)


def recognize(pipeline, img):
    package = Package(img)
    pipeline(package)
    return package


def main():
    video = 'srpv2'

    with open(f'assets/configs/srpv2.json') as f:
        config = json.load(f)

    pipeline = Pipeline('Pipeline', [
        Pipeline('Preprocessor', [
            thumbnail,
            to_grayscale,
            equalize_histogram,
            BilateralFilter(config['diameter'], config['sigma_color'], config['sigma_space']),
            WolfJolionThreshold(config['window_size'], config['k']),
        ]),
        Pipeline('PlateFinder', [
            extract_countours,
            FilterCharacterRatio(config['min_precision'], config['max_precision']),
            FilterCharacterSize(config['min_area'], config['max_area']),
            ClusterDBSCAN(config['eps'], config['min_samples']),
            horizontality_check,
            unwarp_regions,
            extract_characters,
        ])
    ])

    time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    path = f'assets/analysis/{time}/'
    os.mkdir(path)

    # frame_list = generate_frames_from_video(f'assets/videos/{video}.mp4')
    frame_list = get_images_from_dir('/assets/images/')

    for i, frame in enumerate(frame_list):
        try:
            package = recognize(pipeline, frame)
        except RuntimeError as e:
            log.error(e)
        else:
            # save_collage(path + f'{i}_montage.jpg', package.images.values())
            # save_warp_regions(path, i, package.images.items())
            save_isolated_chars(path, i, package.pipeline_data['IsolatedChars'])


def generate_frames_from_video(filename):
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    if ret:
        yield frame
    cap.release()


def get_images_from_dir(path):
    # for filename in sorted(os.listdir(path)):
    for filename in os.listdir(path):
        img_path = path + filename
        img = cv2.imread(img_path)
        yield img


def save_collage(path_name, images):
    def to_bgr(img):
        if len(img.shape) != 3:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    images = list(map(to_bgr, images))
    montage = build_montages(images, (640, 360), (4, 4))[0]
    cv2.imwrite(path_name, montage)


def save_warp_regions(path, i, items):
    for method, img in items:
        if 'UnwarpRegions' in method:
            cv2.imwrite(path + f'{i}_{method}.jpg', img)


def save_isolated_chars(path, i, items):
    for k, plate in enumerate(items):
        for j, char in enumerate(plate):
            cv2.imwrite(path + f'{i}_IsolatedChars_{k}_{j}.jpg', char)


if __name__ == '__main__':
    main()
