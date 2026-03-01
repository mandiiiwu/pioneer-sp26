import cv2
import numpy as np 

cap = cv2.VideoCapture(0)
change = 0 


while True:
    ret, frame = cap.read()
    if not ret: break 

    h, w, _ = frame.shape 

    orig_pts = np.float32([
        [10, 10], [w-10, 10], [w-10, h-10] 
    ])

    if change >= 300: change = 10
    else: change += 10 

    new_pts = np.float32([
        [10, 10], [w+change, 10], [w+change, h+change]
    ])

    matrix = cv2.getAffineTransform(orig_pts, new_pts)
    warped = cv2.warpAffine(frame, matrix, (w, h))

    cv2.imshow('warped', warped)
    # seems to only expand it; im not sure if this was the intended goal though 

    if cv2.waitKey(1) != -1: break 

cv2.destroyAllWindows() 