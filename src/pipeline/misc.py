import logging
from operator import attrgetter

import cv2
import imutils
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from util import Region, Rect, draw_rectangles_on_image

from . import Package

log = logging.getLogger(__name__)


def thumbnail(package: Package):
    package.images['Thumbnail'] = imutils.resize(package.current_image, width=640)


def to_grayscale(package: Package):
    package.images['ToGrayscale'] = cv2.cvtColor(package.current_image, cv2.COLOR_BGR2GRAY)


def extract_countours(package: Package):
    _, contours, hierarchy = cv2.findContours(package.current_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    package.pipeline_data['Contours'] = (contours, hierarchy)


class ClusterDBSCAN:
    def __init__(self, eps=0.5, min_samples=2):
        """
        Returns
        -------
        List[List[Rect]]
            First list is the clusters, and second list is the
            character rectangles that are inside that cluster
        """
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    def __call__(self, package: Package):
        rects = np.array(package.latest_pipeline_data)

        if len(rects) == 0:
            raise RuntimeError('No rects found!')

        if len(rects) < 4:
            raise RuntimeError('Not enough rects!')

        rect_centers = self.calculate_centers(rects)
        # Normalize the points so that DBSCAN's epsilon parameter works
        rect_centers = StandardScaler().fit_transform(rect_centers)

        labels = self.dbscan.fit_predict(rect_centers)
        unique_labels = set(labels)

        clusters = [rects[labels == k] for k in unique_labels if k > -1]
        package.pipeline_data['Clustering'] = clusters

        cp = package.threshold_image
        for i, c in enumerate(package.latest_pipeline_data):
            for rect in c:
                cv2.putText(cp, str(i), (rect.x, rect.y + rect.h), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0xFF), 3,
                            cv2.LINE_AA)
        package.images['Clustering'] = cp

    def calculate_centers(self, rects):
        centers = np.stack(r.center() for r in rects).astype(np.float32)
        return centers


def horizontality_check(package: Package):
    clusters = package.latest_pipeline_data
    clusters = [sorted(cluster, key=attrgetter('x')) for cluster in clusters]

    def horizontal_filter(cluster):
        max_top = max(rect.top for rect in cluster)
        min_bottom = min(rect.bottom for rect in cluster)
        return max_top < min_bottom

    filtered_clusters = list(filter(horizontal_filter, clusters))
    if not filtered_clusters:
        raise RuntimeError('Nothing is horizontal enough')

    package.pipeline_data['HorizontalityCheck'] = filtered_clusters

    regions = [Region(cluster) for cluster in filtered_clusters]
    package.images['Regions'] = draw_rectangles_on_image(package.threshold_image, regions)


def unwarp_regions(package: Package):
    clusters = package.latest_pipeline_data
    img = package.threshold_image

    # fix to be binary again
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[img != 0] = 255

    unwarped = []
    for cluster in clusters:
        top = min(rect.top for rect in cluster)
        bottom = max(rect.bottom for rect in cluster)
        left = min(rect.left for rect in cluster)
        right = max(rect.right for rect in cluster)

        unwarped.append(img[top:bottom, left:right])

    package.pipeline_data['UnwarpRegions'] = unwarped
    for i, img in enumerate(package.latest_pipeline_data):
        package.images[f'UnwarpRegions_{i}'] = img


def unwarp_regions_old(package: Package):
    clusters = package.latest_pipeline_data
    unwarped = [unwarp_region(cluster, package.threshold_image) for cluster in clusters]
    package.pipeline_data['UnwarpRegions'] = unwarped
    for i, img in enumerate(package.latest_pipeline_data):
        package.images[f'UnwarpRegions_{i}'] = img


def unwarp_region(cluster, image):
    # cluster = sorted(cluster, key=attrgetter('x'))
    left, right = cluster[0], cluster[-1]

    actual_coords = np.array([
        left.top_left,
        left.bottom_left,
        right.top_right,
        right.bottom_right,
    ])

    from imutils.perspective import four_point_transform
    transformed = four_point_transform(image, actual_coords)

    # fix to be binary again
    transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
    transformed[transformed != 0] = 255

    return transformed


def extract_characters(package: Package):
    ideal_character_ratio = 50 / 79

    regions = package.latest_pipeline_data

    def extract(region):
        # Remove small specs
        region = cv2.bitwise_not(region)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel, iterations=1)

        _, contours, hierarchy = cv2.findContours(region.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)

        biggest_amount = 7
        biggest = contours[:biggest_amount]
        if len(biggest) < biggest_amount:
            raise RuntimeError('Not enough contours')

        inverted = cv2.bitwise_not(region)

        chars = []
        for cnt in biggest:
            bounds = Rect(cv2.boundingRect(cnt))
            cut = inverted[bounds.top:bounds.bottom, bounds.left:bounds.right]
            chars.append(cut)

        return chars

    list_of_plates = list(map(extract, regions))
    package.pipeline_data['IsolatedChars'] = list_of_plates


def extract_characters_stats(package: Package):
    ideal_character_ratio = 50 / 79

    images = package.latest_pipeline_data

    list_of_plates = []
    for img in images:

        # improve images
        for i in range(2):
            log.debug('filter_biggest_contour %s\n%s', i, img)
            img = morph_close(img)
            img = filter_biggest_contour(img)

        height, width = img.shape
        stride = int(height * ideal_character_ratio)
        wiggle = 5

        def left_stride(i):
            return max(0, stride * i - wiggle)

        def right_stride(i):
            return stride * (i + 1) + wiggle

        # Pegando os 3 caracteres da esquerda
        chars_esq = [img[0:height, left_stride(i):right_stride(i)] for i in range(3)]

        def left_stride(i):
            return width - stride * (i + 1) - wiggle

        def right_stride(i):
            return width - stride * i + wiggle

        # Pegando os 4 numeros da direita
        chars_dir = [img[0:height, left_stride(i):right_stride(i)] for i in reversed(range(4))]

        plate = chars_esq + chars_dir

        list_of_plates.append(plate)

        # package.pipeline_data['IsolatedChars'] = list_of_plates


def morph_close(plate):
    log.debug(plate[0])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = [cv2.morphologyEx(char, cv2.MORPH_CLOSE, kernel) for char in plate]
    return morphed


def filter_biggest_contour(plate):
    filtered_contours = [list(map(contour_filter, char)) for char in plate]
    return filtered_contours


def contour_filter(char):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(char)

    if len(stats) < 2:
        raise RuntimeError('No character in character crop!')

    stats = list(enumerate(stats))[1:]
    stats.sort(key=lambda x: x[1][4], reverse=True)  # sort by area of contour

    largest_index, largest_comp = stats[0]

    bounds = Rect(largest_comp[:4])
    img = labels[bounds.top:bounds.bottom, bounds.left:bounds.right]
    img[img != largest_index] = 255
    img[img == largest_index] = 0

    return img
