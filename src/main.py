import logging

import cv2

from src.recognizer import Recognizer

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

if __name__ == '__main__':

    video = 23

    recognizer = Recognizer(f'assets/configs/{video}.json')

    cap = cv2.VideoCapture(f'assets/videos/{video}.mp4')

    for _ in range(1):
        ret, frame = cap.read()

        package = recognizer.recognize(frame)
        log.info(f'Recognizer output: {package}')

        log.debug(package.pipeline_data)

        for method, img in package.images.items():
            cv2.imshow(method, img)
            cv2.waitKey(0)
            cv2.destroyWindow(method)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        if not ret:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
