import cv2 
import numpy as np 

angle = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break 

    h, w, _ = frame.shape 

    matrix = cv2.getRotationMatrix2D((h//2, w//2), angle, 1)
    rot = cv2.warpAffine(frame, matrix, (w, h))

    cv2.imshow('rotated', rot)
    angle += 5
    
    if cv2.waitKey(1) != -1: break

cv2.destroyAllWindows()