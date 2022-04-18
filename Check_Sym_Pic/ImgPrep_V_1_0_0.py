#!/usr/bin/env python
# coding: utf-8

import cv2
import pathlib
import os
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

ia.seed(42)

# # Готовим дадасет.
# - Делаем изображение чернобелым и удаляем оттенки серого  
# - Создаем для каждого изображения аугментированные копии 30 шт
# - Инвертируем изображение, просто мне так больше нравится
dir_name = input("Введите имя папки с изображениями: ")
n = int(input("Введите количество изображениями: "))

wdir = os.getcwd()
os.chdir(dir_name)
d_set = os.getcwd()
cell = 0
print("теперь тут: " + d_set)
for something in os.listdir():    
    if (os.path.isdir(something)):
        i=1 
        pic = os.getcwd()+"/"+something
        image = imageio.imread(os.getcwd()+"/"+something + "/thumbnail_128x128.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, image = cv2.threshold(image, 20, 255, 0)
        image_bk = image
        for i in range(n):
            seq = iaa.Sequential([
            iaa.Affine(rotate=(-1, 1)),
            iaa.Crop(percent=(0, 0.005))
            ], random_order=True)
            images_aug = seq(image=image)
            ret, images_aug = cv2.threshold(images_aug, 20, 255, cv2.THRESH_BINARY_INV)                
            cv2.imwrite(os.getcwd()+"/"+ something +'/thumbnail_128x128_' + str(i) +'.png', images_aug)
        ret, image = cv2.threshold(image_bk, 20, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(os.getcwd()+"/"+ something +'/thumbnail_128x128_0.png', image)
        os.remove(os.getcwd()+"/"+something +'/thumbnail_128x128.png')
    cell = cell + 1
print("\nКоличество классов: " + str(cell))
