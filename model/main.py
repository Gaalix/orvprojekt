import tensorflow as tf
from keras.src.utils import load_img, img_to_array, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import random

data_dir = '../data'

images = []
labels = []

min_num_images = min(len(os.listdir(os.path.join(data_dir, folder))) for folder in os.listdir(data_dir))

for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    files = os.listdir(folder_path)

    sampled_files = random.sample(files, min_num_images)

    for file in sampled_files:
        file_path = os.path.join(folder_path, file)

        img = load_img(file_path, target_size=(64, 64))
        img_array = img_to_array(img)

        images.append(img_array)
        labels.append(folder)

images = np.array(images)
labels = np.array(labels)

images = images / 255.0

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)


