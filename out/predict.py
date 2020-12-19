import numpy as np
import cv2

import matplotlib.pyplot as plt

from common.constants import TEST_DIR, MODEL_PATH, EMOTION_MAP, IMG_DIM, EMOTION_KEY_MAP, MODEL_PATH_test
from test.results import Result

from keras.models import load_model
from keras.preprocessing import image as kimg
from model import create_model


def predict_emotion(img, m):
    """Returns a prediction of what emotion img contains given the model.

        Args:
            img : ndarray
                The input img (grayscaled).
            m : keras.models
                A keras model.
        Returns:
            prediction : int
                The int that represents the predicted emotion.
            prediction_vector : ndarray
                The prediction vector containing probabilies for each emotion.
        """
    images = []
    img = kimg.img_to_array(img)

    # Since our model takes batches, we make a batch containing copies of img
    img = np.resize(img, (1, 48, 48, 1))
    images.append(img)

    prediction_vector = m.predict(images)
    prediction = np.argmax(prediction_vector, axis=1)

    return prediction, prediction_vector

def display_expression(full_img, model, mode=0, output_file=None):
    """Returns a prediction of what emotion img contains given the model.

        Args:
            full_img : img
                The input img (grayscaled).
            model : keras.models
                A keras model.
            mode : int
                Mode that enables relevant functionality for single image (mode = 0) or video (mode = 1)
            output_file: string/None
                Output file to write instead of show the image
        """
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

def detect_emotions_webcam(model):
    """Turns on webcam and analyses facial expression using given keras CNN model.

        Args:
            m : keras.models
                A keras model.
    """
    webcam = cv2.VideoCapture(0)

    while True:
        capture, frame = webcam.read()

        if capture == False:
            continue

        # in video mode
        display_expression(frame, model, mode=1)

    return

if __name__ == "__main__":    
    # Find facial expression in an individual img
    model = load_model(MODEL_PATH)

    # path = "./Test pics/How-To-Control-Hunger-E28093-20-Best-Strategies-To-Stop-Feeling-Hungry-All-The-Time-624x702.png"
    path = "./Test pics/barry.jpg"
    # path = "./Test pics/got.jpg"

    full_img = cv2.imread(path)
    display_expression(full_img, model)

    #detect_emotions_webcam(model)

    # Test our models accuracy across our testing set
    test = True

    if test == True:
        total = 0
        correct = 0

        for emotion in os.listdir(TEST_DIR):
            result = Result(emotion)
            for filename in os.listdir(os.path.join(TEST_DIR, emotion)):
                path = os.path.join(os.path.join(TEST_DIR,emotion), filename)
                img = kimg.load_img(path, target_size=(IMG_DIM, IMG_DIM), color_mode="grayscale")

                # models prediction
                prediction, pv = predict_emotion(img, model)
                emotion_str = EMOTION_MAP[prediction[0]]

                result.add(emotion_str)

            #result.summarize()
            total += result.total
            correct += result.predictions[EMOTION_KEY_MAP[emotion]]

        print("test accuracy: {}".format(correct/total))
