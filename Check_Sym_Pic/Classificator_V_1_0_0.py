#!/usr/bin/env python
# coding: utf-8



import os
import cv2
import numpy as np
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd


batch_size=30
image_size=(128, 128)

e = int(input("\nВведите количество эпох для обучения нейросети: "))

train_dataset = image_dataset_from_directory('./traning_set',
                                             subset='training',
                                             seed=42,
                                             validation_split=0.1,
                                             batch_size=batch_size,
                                             image_size=image_size,
                                             color_mode="grayscale")

validation_dataset = image_dataset_from_directory('./traning_set',
                                             subset='validation',
                                             seed=42,
                                             validation_split=0.1,
                                             batch_size=batch_size,
                                             image_size=image_size,
                                             color_mode="grayscale")


class_names = train_dataset.class_names

import csv 

with open("class_names.csv", "w", newline='') as file:
    csv.writer(file).writerow(class_names)

test_dataset = image_dataset_from_directory('./test_set',
                                             seed=42, 
                                            batch_size=batch_size,
                                             image_size=image_size,
                                             color_mode="grayscale")

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Создаем последовательную модель
model_f7_5_3_drop = Sequential()
# Сверточный слой
model_f7_5_3_drop.add(Conv2D(16, (7, 7), padding='same', 
                 input_shape=(128, 128, 1), activation='relu'))
# Слой подвыборки
model_f7_5_3_drop.add(MaxPooling2D(pool_size=(2, 2)))
# Сверточный слой
model_f7_5_3_drop.add(Conv2D(32, (5, 5), padding='same', 
                 input_shape=(128, 128, 1), activation='relu'))
# Слой подвыборки
model_f7_5_3_drop.add(MaxPooling2D(pool_size=(2, 2)))
# Сверточный слой
model_f7_5_3_drop.add(Conv2D(64, (3, 3), padding='same', 
                 input_shape=(128, 128, 1), activation='relu'))
# Слой подвыборки
model_f7_5_3_drop.add(MaxPooling2D(pool_size=(2, 2)))
# Полносвязная часть нейронной сети для классификации
model_f7_5_3_drop.add(Flatten())
model_f7_5_3_drop.add(Dense(512, activation='relu'))
model_f7_5_3_drop.add(Dropout(0.1))
# Выходной слой, 208 нейронов по количеству классов
model_f7_5_3_drop.add(Dense(208, activation='softmax'))
model_f7_5_3_drop.summary()

model_f7_5_3_drop.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

history = model_f7_5_3_drop.fit(train_dataset, 
                    validation_data=validation_dataset,
                    epochs=e,
                    verbose=2)

model_f7_5_3_drop.save("Symbol_Check_model_f7_5_3_drop.h5")

scores = model_f7_5_3_drop.evaluate(test_dataset, verbose=1)

print("\nНейросеть сохранена в файл: Symbol_Check_model_f7_5_3_drop.h5")
