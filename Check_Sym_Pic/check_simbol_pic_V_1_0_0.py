import csv
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
import matplotlib.pyplot as plt
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import re 

wdir = os.getcwd()
lib_path = input("\nВведите путь до библиотеки: ")
sfx = input("\nВведите суффикс для библиотеки: ")
#print("сейчас тут: " + wdir)
#os.chdir("./GitHub")
os.chdir(lib_path)
d_set = os.getcwd()
loaded_model = load_model('../Symbol_Check_model_f7_5_3_drop.h5')
all_cells = []
errors = []
k=0
m=0
exeption_cell = np.loadtxt("../exeption.csv", delimiter=",", dtype='str')
class_names = np.loadtxt("../class_names.csv", delimiter=",", dtype='str')
#print("теперь тут: " + d_set)
for cell in os.listdir():     
       if os.path.isdir(cell): 
            k=k+1
            cell_s =re.sub(sfx + "X\d+", "", cell)
            for e_cell in exeption_cell:                
                if re.search(e_cell, cell):
                    cell_s = e_cell            
                if cell_s == "DECAP":
                    cell_s = "FCNE"
                if (cell_s == "STE" or cell_s == "CLKVBUF" or cell_s == "DLY" or cell_s == "BUF"):
                    cell_s = "BU"
                if (cell_s == "ANTENNACELLN2" or cell_s == "ANTENNACELLP2"):
                    cell_s = "ANTENNACELL"    
            #print(cell+"--->"+cell_s)
            image = imageio.imread(os.getcwd()+"/"+cell+"/symbol/thumbnail_128x128.png")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, image = cv2.threshold(image, 20, 255, 0)
            ret, image_inv = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY_INV)
            #ia.imshow(image_inv)
            x = np.array([image_inv])
            prediction = loaded_model.predict(x)
            prediction = np.argmax(prediction)
            if cell_s != class_names[prediction]:                
                print("\nВызывает подозрение ячейка: "+ cell+"--->"+class_names[prediction]+sfx)
                errors.append(cell)
                m = m + 1
my_file = open("../errors.txt", "w")
for  error in errors:
    my_file.write(error + '\n')
my_file.close()
print("\nОбнаружено подозрительных ячеек: "+ str(m))
print("\nИмена ячеек вызывающих подозрение, сохранены в файл: "+ wdir + "\errors.txt")
print("\nВсего ячеек обработано: "+ str(k))




