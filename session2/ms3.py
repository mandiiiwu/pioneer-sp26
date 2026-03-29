import cv2
import numpy as np 

import os 

curr = os.getcwd() 
img = cv2.imread(f'{curr}/SampleImages/chicago.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# sobel 
horz_vals = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
horz_img = cv2.convertScaleAbs(horz_vals)

vert_vals = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
vert_img = cv2.convertScaleAbs(vert_vals)

sobel_final = cv2.convertScaleAbs(cv2.addWeighted(vert_img, 0.5, horz_img, 0.5, 0))

# canny 
blurred = cv2.GaussianBlur(gray, (5, 5), 1)
canny = cv2.Canny(blurred, 25, 150)

cv2.imshow('original', img)

cv2.imshow('horizontal gradient', horz_img)
cv2.imshow('vertical gradient', vert_img)
cv2.imshow('sobel', sobel_final)

cv2.imshow('blurred', blurred)
cv2.imshow('canny', canny)

cv2.waitKey(-1)

cv2.destroyAllWindows()