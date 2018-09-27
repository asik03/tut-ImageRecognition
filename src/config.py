import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SRC_DIR, '..')
ETC_DIR = os.path.join(PROJECT_DIR, 'etc')

MODELS_DIR = os.path.join(ETC_DIR, 'models')

IMAGES_DIR = os.path.join(ETC_DIR, 'images')

TRAIN_DIR = os.path.join(ETC_DIR, 'train')
TEST_DIR = os.path.join(ETC_DIR, 'test')

XML_FACE = os.path.join(ETC_DIR, 'resources', 'haarcascade_frontalface_default.xml')
