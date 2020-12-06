import os
import numpy as np
from constants import TEST_DIR, MODEL_PATH

from keras.models import load_model
from keras.preprocessing import image as kimg



model = load_model(MODEL_PATH)

def folderEncoder(folderName):
    if folderName=='Angry':
        return 0
    elif folderName=="Disgust":
        return 1
    elif folderName=='Fear':
        return 2
    elif folderName=="Happy":
        return 3
    elif folderName=='Neutral':
        return 4
    elif folderName=="Sad":
        return 5
    elif folderName=='Surprise':
        return 6
    else:
        return None


def predict_emotion(img, m):
    images = []
    img = kimg.img_to_array(img)

    # Since our model takes batches, we make a batch containing copies of img
    img = np.resize(img, (1, 48, 48, 1))
    images.append(img)

    prediction = m.predict(images)
    prediction = np.argmax(prediction, axis=1)

    return prediction


result=[]
total = 0
correct = 0
for folder1 in os.listdir(TEST_DIR):
    print(folder1)
    y_true=folderEncoder(folder1)
    for filename in os.listdir(os.path.join(TEST_DIR,folder1)):
        path = os.path.join(os.path.join(TEST_DIR,folder1), filename)
        img = kimg.load_img(path, target_size=(48, 48), color_mode="grayscale")

        # models prediction
        y_pred = predict_emotion(img, model)

        result.append([y_true, y_pred[0]])

        #if it matches true label
        if y_true == y_pred[0]:
            correct+=1

        total += 1

print("test accuracy: {}".format(correct/total))