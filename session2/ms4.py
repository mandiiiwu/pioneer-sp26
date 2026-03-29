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

# hsv thresh turned out a lot cleaner so lets use that

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
final = img.copy() 

if len(contours) > 0:
    ball = max(contours, key=cv2.contourArea)
    cv2.drawContours(final, [ball], -1, (0, 255, 0), 2)

    x, y, w, h = cv2.boundingRect(ball)
    cv2.rectangle(final, (x, y), (x+w, y+h), (0, 0, 255), 2)

    (cx, cy), r = cv2.minEnclosingCircle(ball)
    cv2.circle(final, (int(cx), int(cy)), int(r), (255, 0, 0), 2)

    print(f'largest contour area: {cv2.contourArea(ball)}')
    cv2.imshow('final', final)

    cv2.waitKey(-1)

cv2.destroyAllWindows()

