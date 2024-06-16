import cv2
import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import math
import argparse
import os
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import imagecodecs

def restore_broken_pixels(image):
    interpolated_image = image.resize(image.size, Image.BICUBIC)
    return interpolated_image

def enhance_image(image):
    enhancer = ImageEnhance.Sharpness(image)
    sharp_image = enhancer.enhance(2.0)
    noisy_image = np.array(sharp_image)
    noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR)
    denoised_image = bilateral_filter(noisy_image, 3, 75, 75)  # Параметры можно изменять
    denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
    enhanced_image = Image.fromarray(denoised_image)
    enhanced_image = ImageOps.equalize(enhanced_image)
    return enhanced_image


def bilateral_filter(image, diameter, sigma_color, sigma_space):
    # Применение билатерального фильтра из OpenCV
    bilateral_filtered_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    return bilateral_filtered_image

parser = argparse.ArgumentParser(description="Parser of inputStr")
parser.add_argument("--crop_name", help="Введите название файла", type=str)
parser.add_argument("--layout_name", help="Введите название файла", type=str)

args = parser.parse_args()

# %matplotlib inline

# Путь к файлу .tif
if args.crop_name is not None:
    crop_filepath = args.crop_name
    if args.layout_name is not None:
        tiflayout_filepath = args.layout_name
    else:
        print("Нет файлов")
else:
    print("Нет файлов")


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

    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx


    (x1, y1) = k_1[img1_idx].pt
    (x2, y2) = k_2[img2_idx].pt
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

#print(x, y)

xoffset, px_w, rot1, yoffset, px_h, rot2 = dataset.GetGeoTransform()


posX = px_w * x + rot1 * y + xoffset
posY = rot2 * x + px_h * y + yoffset


posX += px_w / 2.0
posY += px_h / 2.0

print(posX, posY)
tif_filepath = args.crop_name
tif_image = imagecodecs.imread(tif_filepath)
tif_image_pil = Image.fromarray(np.uint8(tif_image))
restored_image = restore_broken_pixels(tif_image_pil)
enhanced_image = enhance_image(restored_image)
output_filename = "tif_new_crop.tif"
enhanced_image.save(output_filename)

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
