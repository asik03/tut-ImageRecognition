import src.config as config
from src.cnn.cnn_smile import CNNSmile
from src.utilities.data_loader import DataLoader

if __name__ == '__main__':
    batch_size = 32
    input_shape = (64, 64, 3)

    train_x, test_x, train_y, test_y = DataLoader.get_train_data(config.IMAGES_DIR, ['smile', 'non_smile'])

    cnn_smile = CNNSmile()
    cnn_smile.build_model(input_shape)

    cnn_smile.train_model(train_x, train_y)
