TRAIN_DIR = r"./train/Datasets/Kaggle/train/"
TEST_DIR = r"./train/Datasets/Kaggle/test/"
VALIDATION_DIR = r"./train/Datasets/Kaggle/validation/"
MODEL_PATH = r"./train/Datasets/model.h5"
MODEL_PATH_test = r"./train/Datasets/model_test.h5"

DATASET_PACKAGE_PATH = r"./train/datasets.gzip"
TEST_PICS_PACKAGE_PATH = r"./test/test_pics.gzip"

IMAGE_UPLOAD_PATH = r"./use/"
IMAGE_OUTPUT_PATH = r"./use/out.jpg"
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

IMG_DIM = 48
BATCH_SIZE = 64
EPOCHS = 50

EMOTION_MAP = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprise"}

# Reverse emotion_map
EMOTION_KEY_MAP = {}
for key in EMOTION_MAP:
    EMOTION_KEY_MAP[EMOTION_MAP[key]] = key