import os
import cv2

current = os.getcwd() + '/session1/'
img_names = os.listdir('/SampleImages')

for name in img_names:
    if (name.endswith('.jpg')):
        img = cv2.imread(f'{current}/SampleImages/{name}')
        cv2.imshow('Images', img)
        cv2.waitKey()

cv2.destroyAllWindows()
