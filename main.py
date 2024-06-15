import cv2
import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import math

# %matplotlib inline

# Путь к файлу .tif
tiflayout_filepath = "18. Sitronics\\18. Sitronics\\layouts\\layout_2021-08-16.tif"
crop_filepath = "18. Sitronics\\18. Sitronics\\1_20\\crop_1_0_0000.tif"

# Открытие файла в режиме чтения
dataset = gdal.Open(tiflayout_filepath, gdal.GA_ReadOnly)
crop = gdal.Open(crop_filepath, gdal.GA_ReadOnly)

# Проверка открытия файла
if dataset is None:
    print("Ошибка открытия файла:", tiflayout_filepath)
    exit(1)

# Доступ к растровым данным
band = dataset.GetRasterBand(1)
data = band.ReadAsArray()
crop_band = crop.GetRasterBand(1)
crop_data = crop_band.ReadAsArray()

data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
data = cv2.resize(data, (5000, 5000))
crop_data = cv2.normalize(crop_data, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

sift = cv2.SIFT_create()

k_2, des_2 = sift.detectAndCompute(crop_data, None)
k_1, des_1 = sift.detectAndCompute(data, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(des_1, des_2)
matches = sorted(matches, key=lambda x: x.distance)

list_kp1 = list()
list_kp2 = list()

for mat in matches:

    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx

    # x - columns
    # y - rows
    # Get the coordinates
    (x1, y1) = k_1[img1_idx].pt
    (x2, y2) = k_2[img2_idx].pt

    # Append to each list
    list_kp1.append((x1, y1))
    list_kp2.append((x2, y2))

x = 0
y = 0
sx = 0
sy = 0
for i in range(len(list_kp1)):
    sx += list_kp1[i][0]
    sy += list_kp1[i][1]
x = sx / len(list_kp1)
y = sy / len(list_kp1)

print(x, y)

xoffset, px_w, rot1, yoffset, px_h, rot2 = dataset.GetGeoTransform()

# supposing x and y are your pixel coordinate this
# is how to get the coordinate in space.
posX = px_w * x + rot1 * y + xoffset
posY = rot2 * x + px_h * y + yoffset

# shift to the center of the pixel
posX += px_w / 2.0
posY += px_h / 2.0

print(posX, posY)

img3 = cv2.drawMatches(data, k_1, crop_data, k_2, matches[:50], crop_data, flags=2)
img3 = cv2.resize(img3, (1000, 900))
cv2.imshow("Output", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Закрытие файла
dataset.FlushCache()
dataset = None
crop.FlushCache()
crop = None
