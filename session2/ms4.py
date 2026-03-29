import cv2
import numpy as np
import os

img = cv2.imread(f'{os.getcwd()}/BallFinding/Green/Green1BG1Near.jpg')
blurred = cv2.GaussianBlur(img, (9, 9), 1)

# using hsv thresholding
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

lower = np.array([36, 25, 80])
upper = np.array([86, 255, 255])

mask = cv2.inRange(hsv, lower, upper)
hsv_thresh = cv2.bitwise_and(img, img, mask=mask)

# using canny edge 
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 100, 200)

cv2.imshow('original', img)
cv2.imshow('blurred', blurred)

cv2.imshow('hsv thresholding', hsv_thresh)
cv2.imshow('canny edge', canny)

cv2.waitKey(-1)

cv2.destroyAllWindows()

