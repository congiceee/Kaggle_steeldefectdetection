import numpy as np
import pandas as pd


train_df = pd.read_csv("data/train.csv")
sample_df = pd.read_csv("data/sample_submission.csv")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

image_0 = 'data/train_images/0a1cade03.jpg'
img_path = image_0

img = mpimg.imread(img_path)
print(img.shape)
print(img)
img = img.astype(np.float32) / 255.
print(img)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread(img_path)
gray = rgb2gray(img)
plt.imshow(gray, cmap='gray')
plt.show()

imgcv2 = cv2.imread(img_path)

print(img.shape)
print(gray.shape)