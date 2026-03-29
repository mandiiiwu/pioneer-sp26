import os
import cv2

img = cv2.imread(f'{os.getcwd()}/SampleImages/snow.jpg')
# cv2.imshow('image', img)
# cv2.waitKey() 

h, w, _ = img.shape # 768, 1024 
cen_x, cen_y = w//2, h//2

# head
cv2.circle(img, center=(cen_x, cen_y), radius=50, color=(0, 0, 0), thickness=3) 

# torso!
cv2.line(img, (cen_x, cen_y+50), (cen_x, cen_y+150), color=(0,0,0,), thickness=3) 

# arms
cv2.line(img, (cen_x, cen_y+50), (cen_x-50, cen_y+100), color=(0,0,0,), thickness=3) 
cv2.line(img, (cen_x, cen_y+50), (cen_x+50, cen_y+100), color=(0,0,0,), thickness=3) 

# legs
cv2.line(img, (cen_x, cen_y+150), (cen_x-50, cen_y+225), color=(0,0,0), thickness=3)
cv2.line(img, (cen_x, cen_y+150), (cen_x+50, cen_y+225), color=(0,0,0), thickness=3)

cv2.imshow('image',img)
# stick figure complete!

cv2.waitKey()