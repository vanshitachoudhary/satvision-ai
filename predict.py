import cv2
import numpy as np

def predict_cloud_mask(image_np):

    hsv = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2HSV
    )

    lower = np.array([0, 0, 180])
    upper = np.array([255, 60, 255])

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask