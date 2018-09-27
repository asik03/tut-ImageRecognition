from src.utilities.data_loader import DataLoader
from src.cnn.cnn_smile import CNNSmile

import src.config as config

if __name__ == '__main__':
    batch_size = 32
    input_shape = (64, 64, 3)

    train_generator = DataLoader.get_train_generator(config.TRAIN_DIR,
                                                     ['non_smile', 'smile'],
                                                     batch_size)
    cnn_smile = CNNSmile()
    cnn_smile.build_model(input_shape)

    cnn_smile.train_generator_model(train_generator, epochs=20)
