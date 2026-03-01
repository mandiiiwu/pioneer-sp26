import os
import cv2

img1 = cv2.imread(f'{os.getcwd()}/session1/SampleImages/daylilies.jpg')
img2 = cv2.imread(f'{os.getcwd()}/session1/SampleImages/astilbe.jpg')

# .shape -> h, w, c 
print(img1.shape) # (480, 640, 3)
print(img2.shape) # (426, 640, 3)
# different dims, so crop rois of both for blending

roi1 = img1[100:420, 100:420]
roi2 = img2[100:420, 100:420]

wgt = 1
while wgt >= 0:
    final = cv2.addWeighted(roi1, wgt, roi2, 1-wgt, 0)
    cv2.imshow('blended image', final)
    cv2.waitKey()
    wgt-=0.05

cv2.destroyAllWindows()