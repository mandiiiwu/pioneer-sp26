import cv2
import os 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # extension 1 
    mirrored = cv2.flip(frame, 1)
    cv2.imshow('frame', mirrored) 

    char = cv2.waitKey(1)
    if char == -1: continue
    else:
        char = chr(char)
        if char == 'q': break 
        # extension 2
        if char == ' ': cv2.imwrite(f'/session1/screenshot.jpg', mirrored)

cap.release()