import cv2
import numpy as np 

import os 

curr = os.getcwd() 
img = cv2.imread(f'{curr}/BallFinding/Blue/Blue1BG1Near.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('image', img) # ball appears to be bright blue 

# source: https://www.geeksforgeeks.org/python/filter-color-with-opencv/ 
# blue mask used in source code: [60, 35, 140] (lower) & [180, 255, 255] (upper)

# original mask produced bad results
# lots of manual tweaking later, i found that these values were ideal 
# bit of black corner (bottom left) still persists though 
lower = np.array([100, 100, 20])
upper = np.array([130, 255, 255])

mask = cv2.inRange(hsv, lower, upper)

result = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('og', img)
cv2.imshow('hsv filtered', result)

cv2.waitKey(-1)

cv2.destroyAllWindows()