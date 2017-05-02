import logging

import cv2
import imutils
import numpy as np

from src.pipeline.filter import SizeFilter, CharacterRatioFilter
from . import Pipeline, Package

log = logging.getLogger(__name__)


class GrayscaleThumbnail(Pipeline):
    def __init__(self):
        super().__init__(
            Thumbnail(),
            ToGrayscale(),
        )


class Thumbnail(Pipeline):
    def __init__(self):
        super().__init__()

    def apply(self, package: Package):
        package.images['Thumbnail'] = imutils.resize(package.current_image, width=640)


class ToGrayscale(Pipeline):
    def __init__(self):
        super().__init__()

    def apply(self, package: Package):
        package.images['ToGrayscale'] = cv2.cvtColor(package.current_image, cv2.COLOR_BGR2GRAY)


class PlateFinder(Pipeline):
    """
    Takes in an image and returns an iterable of sub-images that
    may be a plate.
    """

    def __init__(self):
        super().__init__(
            Contours(),
            CharacterRatioFilter(),
            SizeFilter(),
            # TODO replace by DBSCAN
            DummyClusterer(),
            UnwarpRegions(),
        )


class Contours(Pipeline):
    """
    Should only be called via CharacterContours.
    Basic application of findContours().
    """

    def __init__(self):
        super().__init__()

    def apply(self, package: Package):
        _, contours, hierarchy = cv2.findContours(package.current_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        package.pipeline_data['Contours'] = (contours, hierarchy)


class KMeansClusterer(Pipeline):
    """
    Returns
    -------
    List[List[Rect]]
        First list is the clusters, and second list is the
        character rectangles that are inside that cluster
    """

    def __init__(self):
        super().__init__()

    def apply(self, package: Package):
        rect_centers = self.calculate_centers(package)

        # TODO maybe have it start with an initial label on the center of the image
        compactness, labels, centers = cv2.kmeans(
            data=rect_centers,
            K=1,
            bestLabels=None,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.7),
            attempts=10,
            flags=cv2.KMEANS_RANDOM_CENTERS)

        package.pipeline_data['KMeansClusterer'] = list(
            filter(lambda cluster: len(cluster) > 5, rect_centers[labels.ravel() == 0]))

        # fig, ax = plt.subplots()
        # ax.scatter(cluster[:, 0], cluster[:, 1], c='r')
        # ax.scatter(centers[:, 0], centers[:, 1], s=80, c='y', marker='s')
        # fig.show()

        log.debug(f'clusters: {package.latest_pipeline_data}')
        cp = package.original_image
        for i, c in enumerate(package.latest_pipeline_data):
            for point in c:
                cv2.putText(cp, str(i), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        package.images['KMeansClusterer'] = cp

    def calculate_centers(self, package):
        centers = np.stack(r.center() for r in package.latest_pipeline_data).astype(np.float32)
        log.debug(f' Center of rectangles: {centers}')
        return centers


class DummyClusterer(Pipeline):
    """
    Returns
    -------
    List[List[Rect]]
        First list is the clusters, and second list is the
        character rectangles that are inside that cluster
    """

    def __init__(self):
        super().__init__()

    def apply(self, package: Package):
        package.pipeline_data['DummyClusterer'] = [package.latest_pipeline_data]
        cp = package.original_image
        for i, c in enumerate(package.latest_pipeline_data):
            for rect in c:
                cv2.putText(cp, str(i), (rect.x, rect.y + rect.h), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1,
                            cv2.LINE_AA)
        package.images['DummyClusterer'] = cp


class UnwarpRegions(Pipeline):
    def __init__(self):
        super().__init__()

    def apply(self, package: Package):
        clusters = package.latest_pipeline_data
        unwarped = [self.unwarp_region(cluster, package.images['Threshold']) for cluster in clusters]
        package.pipeline_data['UnwarpRegions'] = unwarped
        for i, img in enumerate(package.latest_pipeline_data):
            package.images[f'UnwarpRegions_{i}'] = img

    def unwarp_region(self, cluster, image):
        cluster.sort(key=lambda rect: rect.x)
        left, right = cluster[0], cluster[-1]

        actual_coords = np.float32([[
            left.top_left,
            left.bottom_left,
            right.top_right,
            right.bottom_right,
        ]])

        desired_width = right.bottom_right[0] - left.x
        desired_height = max(left.h, right.h)

        desire_coords = np.float32([[
            (0, 0),
            (0, desired_height),
            (desired_width, 0),
            (desired_width, desired_height),
        ]])

        M = cv2.getPerspectiveTransform(actual_coords, desire_coords)
        transformed = cv2.warpPerspective(image, M, (desired_width, desired_height))
        return transformed
