import cv2
import os 

cap = cv2.VideoCapture(f'{os.getcwd()}/BallFinding/Blue/Blue1BG1.avi')
ret, prev = cap.read() 

while True:
    ret, curr = cap.read()
    if not ret: break 

    diff = cv2.absdiff(prev, curr)
    cv2.imshow('motion', diff)

    prev = curr 

    cv2.waitKey(1)

cv2.destroyAllWindows() 

