# -*- coding: utf-8 -*-
import os
from shutil import copy
import cv2 as cv


from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import load_img


class DataLoader:
    @classmethod
    def get_train_generator(cls, data_path, classes, batch_size):
        data_gen = ImageDataGenerator(rescale=1.0/255)

        train_path = os.path.join(data_path, 'train')

        train_generator = data_gen.flow_from_directory(
                train_path,
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
                img = cv.imread(file_path)


                copy(img, new_path)