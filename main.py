import cv2
import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import math
import argparse
import os
from multiprocessing import Pool
import numpy as np
from skimage import io, restoration, exposure, filters
from skimage.transform import resize

def process_image(file_name):
    input_folder = '/Users/ivanyudin/Downloads/IT проекты/pythonProject/data/full_crop'
    output_folder = '/Users/ivanyudin/Downloads/IT проекты/pythonProject/data/tif_new_full_crop'
    image_path = file_name
    image = io.imread(image_path)
    image_restored = resize(image, output_shape=image.shape, order=3)  # order=3 для бикубической интерполяции
    image_sharpened = filters.unsharp_mask(image_restored)
    if image_sharpened.ndim == 3 and image_sharpened.shape[-1] == 4:
        image_denoised = restoration.denoise_bilateral(image_sharpened[..., :3], sigma_color=0.05, sigma_spatial=15,
                                                       channel_axis=-1)
        image_denoised = np.dstack((image_denoised, image_sharpened[..., 3]))
    else:
        image_denoised = restoration.denoise_bilateral(image_sharpened, sigma_color=0.05, sigma_spatial=15,
                                                       channel_axis=-1)
    image_eq = exposure.equalize_hist(image_denoised)
    image_retinex = exposure.adjust_log(image_eq)
    output_path = 'tif_new_crop.tif'
    io.imsave(output_path, image_retinex)

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
process_image(crop_filepath)

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
