import cv2
import numpy as np 

cap = cv2.VideoCapture(0)
k = 1
blur_dir = 2 

while True:
    ret, frame = cap.read()
    if not ret: break 

    # gaussian blurring gradually 

    blurred = cv2.GaussianBlur(frame, (k, k), 0)

    if not (1 <= k + blur_dir <= 50):
        blur_dir*=-1 
    
    k += blur_dir
    # print(k)

    # morphing w/ changing neighborhood sizes 
    # reuse k values since kernel rules r fulfilled 

    closed = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, 
                              np.ones((k, k), np.uint8))

    # display k values on both frames

    cv2.putText(blurred, f'k value: {k}', (30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(closed, f'k value: {k}', (30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    cv2.imshow('blurred', blurred)
    cv2.imshow('morphed', closed)

    if cv2.waitKey(1) != -1: break

cv2.destroyAllWindows()