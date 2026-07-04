import cv2
import numpy as np

image = cv2.imread("data/input/test.jpg")

if image is None:
    print("Image not found!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

result = cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA)

cv2.imwrite("outputs/cloud_mask.png", mask)
cv2.imwrite("outputs/cloud_removed.jpg", result)

print("SUCCESS")