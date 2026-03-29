import os
import cv2

current = os.getcwd()

img_names = [
    'chicago.jpg',
    'daylilies.jpg',
    'astilbe.jpg',
    'snow.jpg'
]

for name in img_names:
    img = cv2.imread(f'{current}/SampleImages/{name}')
    cv2.imshow(f'image', img)
    cv2.waitKey(0)

    