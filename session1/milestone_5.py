import cv2 
import numpy as np
import random 


offset = 0
change = 5

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

    if not (0 <= offset <= 100) or not (0 <= offset <= 100):
        change *= -1 

    offset += change
    
    matrix2 = np.float32([[1, 0, offset],
                         [0, 1, offset]])
    
    bounce = cv2.warpAffine(copy2, matrix2, (w, h))
    
    cv2.imshow('jitter', jitter)
    cv2.imshow('bounce', bounce)

    if cv2.waitKey(1) != -1:
        break

cv2.destroyAllWindows