import os
import numpy as np
import cv2
from constants import TEST_DIR, MODEL_PATH, EMOTION_MAP, IMG_DIM

from keras.models import load_model
from keras.preprocessing import image as kimg


def predict_emotion(img, m):
    images = []
    img = kimg.img_to_array(img)

    # Since our model takes batches, we make a batch containing copies of img
    img = np.resize(img, (1, 48, 48, 1))
    images.append(img)

    prediction = m.predict(images)
    prediction = np.argmax(prediction, axis=1)

    return prediction

def display_expression(full_img, model, mode=0):
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
        expression = predict_emotion(resized_image, model)

        # Label and display image
        cv2.putText(full_img, EMOTION_MAP[expression[0]], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.imshow("Facial Expression Analysis", full_img)
        cv2.waitKey(mode)

        return

    print("Couldn't find face")
    return

def detect_emotions_webcam(model):
    webcam = cv2.VideoCapture(0)

    while True:
        capture, frame = webcam.read()

        if capture == False:
            continue

        # in video mode
        display_expression(frame, model, mode=1)

    return

# Find facial expression in an individual img
model = load_model(MODEL_PATH)
path = "./Test pics/How-To-Control-Hunger-E28093-20-Best-Strategies-To-Stop-Feeling-Hungry-All-The-Time-624x702.png"
full_img = cv2.imread(path)
display_expression(full_img, model)

#detect_emotions_webcam(model)


# Test our models accuracy across our testing set
test = False

if test == True:
    result=[]
    total = 0
    correct = 0

    for emotion in os.listdir(TEST_DIR):
        sub_total = 0
        sub_correct = 0
        for filename in os.listdir(os.path.join(TEST_DIR, emotion)):
            path = os.path.join(os.path.join(TEST_DIR,emotion), filename)
            img = kimg.load_img(path, target_size=(IMG_DIM, IMG_DIM), color_mode="grayscale")

            # models prediction
            prediction = predict_emotion(img, model)
            emotion_str = EMOTION_MAP[prediction[0]]

            #if it matches true label
            if emotion == emotion_str:
                correct+=1
                sub_correct += 1

            total += 1
            sub_total += 1
        print(emotion)
        print("test accuracy: {}".format(sub_correct / sub_total))

    print("test accuracy: {}".format(correct/total))