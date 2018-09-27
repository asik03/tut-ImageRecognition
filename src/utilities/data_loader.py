# -*- coding: utf-8 -*-
import os
from shutil import copy

import cv2 as cv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from src.utilities.face_segmenter import FaceSegmenter


class DataLoader:
    @classmethod
    def get_train_data(cls, data_path, classes):
        data = []
        labels = []
        for class_name in classes:
            class_path = os.path.join(data_path, class_name)
            file_names = os.listdir(class_path)

            file_names = list(filter(lambda x: x.endswith('.jpg'), file_names))
            file_names.sort()

            for file_name in file_names:
                file_path = os.path.join(class_path, file_name)
                image = cv.imread(file_path)
                image = cv.resize(image, (64, 64))
                # image = image.img_to_array()
                data.append(image)
                label = 1 if class_name == 'smile' else 0
                labels.append(label)

        data = np.array(data, dtype='float') / 255.0
        labels = np.array(labels)
        (train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.2, random_state=42)

        # test_y = to_categorical(test_y, num_classes=2)
        # train_y = to_categorical(train_y, num_classes=2)

        return train_x, test_x, train_y, test_y

    @classmethod
    def get_train_generator(cls, data_path, classes, batch_size):
        data_gen = ImageDataGenerator(rescale=1.0/255)

        train_generator = data_gen.flow_from_directory(
                data_path,
                target_size=(64, 64),
                batch_size=batch_size,
                class_mode='binary',
                classes=classes,
                shuffle=True)

        return train_generator

    @classmethod
    def organize_data(cls, etc_path):
        data_path = os.path.join(etc_path, 'raw_data')
        filenames = os.listdir(data_path)

        filenames = list(filter(lambda x: x.endswith('.jpg'), filenames))
        filenames.sort()

        number_smiles = 2162
        number_smiles_train = 1600
        number_non_smiles_train = 1600

        smile_paths = []
        non_smile_paths = []
        for counter, filename in enumerate(filenames):
            file_path = os.path.join(data_path, filename)
            if counter < number_smiles:
                smile_paths.append(file_path)
            else:
                non_smile_paths.append(file_path)

        smile_train_paths = smile_paths[0:number_smiles_train]
        smile_test_paths = smile_paths[number_smiles_train::]

        non_smile_train_paths = non_smile_paths[0:number_non_smiles_train]
        non_smile_test_paths = non_smile_paths[number_non_smiles_train::]

        new_paths = [os.path.join(etc_path, 'train\\smile'),
                     os.path.join(etc_path, 'train\\non_smile'),
                     os.path.join(etc_path, 'test\\smile'),
                     os.path.join(etc_path, 'test\\non_smile')]
        old_paths = [smile_train_paths, non_smile_train_paths, smile_test_paths, non_smile_test_paths]

        for new_path, old_path in zip(new_paths, old_paths):
            for file_path in old_path:
                image = cv.imread(file_path)

                img_cropped_list, _ = FaceSegmenter.segment(image)

                if len(img_cropped_list) == 0:
                    copy(file_path, new_path)
                else:
                    cv.imwrite(os.path.join(new_path, os.path.basename(file_path)), img_cropped_list[0])
