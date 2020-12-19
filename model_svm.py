import numpy as np
import cv2
from sklearn.svm import SVC
from constants import TRAIN_DIR, VALIDATION_DIR, MODEL_PATH, IMG_DIM, BATCH_SIZE, EPOCHS, EMOTION_KEY_MAP, EMOTION_MAP
import os
from keras.preprocessing import image as kimg
import matplotlib.pyplot as plt
import pickle
import descriptors as dsc
import dlib


def get_x_and_y(DIR_NAME, features=None):
    """Returns a ndarray of data/features and its correponding ndarray of labels.

        Args:
            DIR_NAME : str
                The directory name containing folders that represent 7 emotions.
            features : str
                The features we want to extract.
        Returns:
            x : ndarray
                The data/features.
            y : ndarray
                The respective labels for each item in x_train.
        """

    x_train = []
    y_train = []
    shape_predictor = dlib.shape_predictor(r'./shape_predictor.dat')
    img = []

    for emotion in os.listdir(DIR_NAME):
        for filename in os.listdir(os.path.join(DIR_NAME, emotion)):
            path = os.path.join(os.path.join(DIR_NAME,emotion), filename)

            # Use raw pixels as features
            if features is None:
                img = np.array(kimg.load_img(path, target_size=(IMG_DIM, IMG_DIM), color_mode="grayscale"), 'float32').flatten()

            # Use facial landmarks as features
            elif features == 'FL':
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                FL = dsc.get_facial_landmarks(gray, shape_predictor)
                img = FL

            # Use local binary patterns as features
            elif features == 'LBP':
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                LBP = dsc.get_LBP(gray)
                img = LBP

            # Use HOG as features
            elif features == 'HOG':
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                HOG = dsc.get_hog(gray, 8).flatten()
                img = HOG

            # Use combined features
            elif features == 'HOGFL':
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                FL = dsc.get_facial_landmarks(gray, shape_predictor)
                LBP = dsc.get_LBP(gray)
                HOG = dsc.get_hog(gray, 8).flatten()

                img = np.concatenate([FL, HOG])

            x_train.append(img)
            y_train.append(EMOTION_KEY_MAP[emotion])

    x = np.array(x_train, 'float32')
    y = np.array(y_train, 'float32')
    return x, y

# Load data
x_train, y_train = get_x_and_y(r"./Datasets/Kaggle/train", 'HOGFL')
x_test, y_test = get_x_and_y(r"./Datasets/Kaggle/test", 'HOGFL')
print("Finished loading images")

# Create SVM model
model = SVC(random_state=0, max_iter=15000, kernel='rbf', decision_function_shape='ovr', gamma='auto')

training = False

if training == True:
    print("Starting SVM training...")
    model.fit(x_train, y_train)

    #Save model
    print("Saving model")
    with open(r"./Datasets/model_svm.h5", 'wb') as f:
        pickle.dump(model, f)

#Load model
print("Loading model")
model = pickle.load(open(r"./Datasets/model_svm.h5", 'rb'))

testing = True

#Test model
if testing == True:
    predicted = model.predict(x_test)

    correct = 0
    for i in range(len(predicted)):
        if y_test[i] == predicted[i]:
            correct += 1

    print("Accuracy: {}".format(correct/len(predicted)))





