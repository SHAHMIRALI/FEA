import numpy as np
import cv2

import matplotlib.pyplot as plt
from keras.preprocessing import image as kimg

from common.constants import EMOTION_MAP, IMG_DIM


def predict_emotion(img, m):
    images = []
    img = kimg.img_to_array(img)

    # Since our model takes batches, we make a batch containing copies of img
    img = np.resize(img, (1, 48, 48, 1))
    images.append(img)

    prediction_vector = m.predict(images)
    prediction = np.argmax(prediction_vector, axis=1)

    return prediction, prediction_vector

def display_expression(full_img, model, mode=0, output_file=None):
    full_img_g = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)

    # Use haarcascade to find face in img
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(full_img_g, 1.3, 5)

    # Go over all faces found in image and detect emotion
    for (x, y, w, h) in faces:

        # Draw bounding box on img
        full_img = cv2.rectangle(full_img, (x, y), (x + w, y + h), (0, 0, 0), 2)

        # Crop out bounding box
        full_img_bb = full_img[y:y + h, x:x + w]

        # Resize image to 48x48 for our model
        resized_image = cv2.resize(full_img_bb, (IMG_DIM, IMG_DIM))
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        if mode == 0:
            cv2.imwrite("./Test pics/" + str(w) + str(h) + '_faces.jpg', resized_image)

        # Pass image to model
        expression, pv = predict_emotion(resized_image, model)

        # Label and display image
        cv2.putText(full_img, EMOTION_MAP[expression[0]], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        cv2.imshow("Facial Expression Analysis", full_img)
        cv2.waitKey(mode)

        if output_file:
            cv2.imwrite(output_file, full_img)

        # Show prediction vector bar graph
        if mode == 0:
            emotions = []
            probabilities = []
            for i in range(len(pv[0])):
                emotions.append(EMOTION_MAP[i])
                probabilities.append(pv[0][i])

            plt.style.use('ggplot')

            x_pos = [i for i, _ in enumerate(emotions)]

            plt.bar(x_pos, probabilities, color='green')
            plt.xlabel("Emotions")
            plt.ylabel("Probability")
            plt.title("Prediction Vector")

            plt.xticks(x_pos, emotions)

            plt.show()

        return

    print("Couldn't find face")
    return
