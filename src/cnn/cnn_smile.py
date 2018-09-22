# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


class CNNSmile:
    def __init__(self, dropout_prob=0.25):
        self.dropout_prob = dropout_prob
        self.model = None
    
    def build_model(self, input_shape):
        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout_prob))
        
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout_prob))
        
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout_prob))
        
        model.add(Flatten())
        
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout_prob))
        
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        optimizer = keras.optimizers.Adam()
        
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        self.model = model
    
    def train_model(self, x_train, y_train, 
                    batch_size=32, epochs=50):        
        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       shuffle=True)
        
    def train_generator_model(self, train_generator,
                              epochs=50):
        self.model.fit_generator(train_generator,
                                 epochs=epochs)
    
    def evaluate_model(self, x_test, y_test):
        scores = self.model.evaluate(x_test, y_test, 
                                     verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
    
    def save_model(self, model_path):
        self.model.save(model_path)
