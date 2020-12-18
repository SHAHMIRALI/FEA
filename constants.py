TRAIN_DIR = r"./Datasets/Kaggle/train/"
TEST_DIR = r"./Datasets/Kaggle/test/"
VALIDATION_DIR = r"./Datasets/Kaggle/validation/"
MODEL_PATH = r"./Datasets/model.h5"
MODEL_PATH_test = r"./Datasets/model_test.h5"
IMG_DIM = 48
BATCH_SIZE = 64
EPOCHS = 50

EMOTION_MAP = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprise"}

# Reverse emotion_map
EMOTION_KEY_MAP = {}
for key in EMOTION_MAP:
    EMOTION_KEY_MAP[EMOTION_MAP[key]] = key