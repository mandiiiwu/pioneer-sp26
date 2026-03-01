import cv2 
import numpy as np
import random 


offset_x, offset_y = 0, 0
change_x, change_y = 5, 5

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break 

    h, w, _ = frame.shape
    cv2.imshow('original', frame)

    copy = frame.copy() 
    copy2 = frame.copy() 

    rand1 = random.randint(30, 70)
    rand2 = random.randint(30, 70)

    matrix1 = np.float32([[1, 0, rand1],
                         [0, 1, rand2]])
    
    jitter = cv2.warpAffine(copy, matrix1, (w, h))

    if not (0 <= offset_x <= 100) or not (0 <= offset_y <= 100):
        change_x *= -1 
        change_y *= -1 

    offset_x += change_x
    offset_y += change_y 

    matrix2 = np.float32([[1, 0, offset_x],
                         [0, 1, offset_y]])
    
    bounce = cv2.warpAffine(copy2, matrix2, (w, h))
    
    cv2.imshow('jitter', jitter)
    cv2.imshow('bounce', bounce)

    if cv2.waitKey(1) != -1:
        break

cv2.destroyAllWindows