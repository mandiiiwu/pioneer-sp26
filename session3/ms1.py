import os
import random
import time 

import cv2
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

data_img = cv2.imread(f'{os.getcwd()}/session3/digits.png')
gray = cv2.cvtColor(data_img, cv2.COLOR_BGR2GRAY)
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

x = np.array(cells)
print(x.shape) # (50, 100, 20, 20) 

y = np.zeros((50, 100), np.float32)
for dig in range(10):
    row = dig * 5 
    y[row:row+5] = dig 

print(y.shape) # (50, 100) 

x = x.reshape(-1, x.shape[2], x.shape[3])
y = y.flatten() 

print(x.shape, y.shape) # (5000, 20, 20) (5000,)

x = x.reshape(-1, 400).astype(np.float32)
print(x.shape, y.shape) # (5000, 400) (5000,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=42)

knn = cv2.ml.KNearest_create()
knn.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

_, result, neighbors, dist = knn.findNearest(x_test, 3) # k=3
print('result: ', result)
print('neighbors: ', neighbors)
print('distances: ', dist)

matches = result.flatten() == y_test.flatten() 
correct = np.count_nonzero(matches)
acc = correct * 100.0 / result.size 
print('accuracy: ', acc)