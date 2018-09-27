from src.utilities.data_loader import DataLoader

import src.config as config


if __name__ == '__main__':
    DataLoader.organize_data(config.ETC_DIR)
