import os
import src.config as config
from src.cnn.cnn_smile import CNNSmile
from src.utilities.data_loader import DataLoader

if __name__ == '__main__':
    batch_size = 32
    input_shape = (64, 64, 3)

    print('Starting')
    train_x, test_x, train_y, test_y = DataLoader.get_train_data(config.TRAIN_DIR, ['smile', 'non_smile'])
    print('Data loaded.')

    cnn_smile = CNNSmile()
    cnn_smile.build_model(input_shape)

    print(train_x.shape)

    cnn_smile.train_model(train_x, train_y, epochs=50)
    cnn_smile.evaluate_model(test_x, test_y)
    cnn_smile.save_model(os.path.join(config.MODELS_DIR, 'model.h5'))
