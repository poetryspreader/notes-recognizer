import cv2
import numpy as np
from numpy import linalg

from config import *


def get_clef(image, staff):
    """
    Gets the clef from the first staff.

    :param image: image to get the clef from
    :param staff: First staff from the image.
    :return:
    """
    i = 0
    width = image.shape[0]

    window_width = int(2 / 5 * (staff.max_range - staff.min_range))

    up = staff.lines_location[0] - window_width
    down = staff.lines_location[-1] + window_width
    key_width = int((down - up) / 1.3)
    while True:
        window = image[up:down, i:i + key_width]
        if window.sum() / window.size < int(255 * WHITE_PIXELS_PERCENTAGE):
            break
        if i + key_width > width:
            print("No key detected!")
            break
        i += int(key_width / WINDOW_SHIFT)

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/7clef.png", window)
    return window, i, up, down, key_width


def classify_clef(image, staff):
    """
    Classify the clef - violin or bass.

    :return: A string indicating the clef
    """
    original_clef, _, _, _, _ = get_clef(image, staff)

    # Настройка параметров детектора пятен
    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByConvexity = True
    params.minConvexity = 0.3

    # Создание детектора пятен с указанными параметрами
    detector = cv2.SimpleBlobDetector_create(params)

    # Обнаружение кругов
    keypoints = detector.detect(original_clef)

    if len(keypoints) == 0:
        return "violin"
    else:
        return "bass"
