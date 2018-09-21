# -*- coding: utf-8 -*-
import os

from keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    def __init__(self, data_path, classes):
        self.data_path = data_path
        self.classes = classes
        
    def get_train_generator(self, batch_size):
        data_gen = ImageDataGenerator(rescale=1.0/255)
        
        train_path = os.path.join(self.data_path, 'train')
        
        train_generator = data_gen.flow_from_directory(
                train_path,
                batch_size=batch_size,
                class_mode='binary',
                classes=self.classes,
                shuffle=True)
        
        return train_generator
    
    