import os

from common.constants import TEST_DIR, MODEL_PATH, EMOTION_MAP, IMG_DIM, EMOTION_KEY_MAP
from out.predict import predict_emotion
from test.results import Result

from keras.models import load_model
from keras.preprocessing import image as kimg

# Test our models accuracy across our testing set
def test_model(model):
    total = 0
    correct = 0

    for emotion in os.listdir(TEST_DIR):
        result = Result(emotion)
        for filename in os.listdir(os.path.join(TEST_DIR, emotion)):
            path = os.path.join(os.path.join(TEST_DIR,emotion), filename)
            img = kimg.load_img(path, target_size=(IMG_DIM, IMG_DIM), color_mode="grayscale")

            # models prediction
            prediction = predict_emotion(img, model)
            emotion_str = EMOTION_MAP[prediction[0]]

            result.add(emotion_str)

        # Print results for each emotion
        result.summarize()

        # Store the results for final summary
        total += result.total
        correct += result.predictions[EMOTION_KEY_MAP[emotion]]

    # Final summary
    print("test accuracy: %.2f" % (correct/total))

if __name__ == "__main__":
    # Find facial expression in an individual img
    model = load_model(MODEL_PATH)

    test_model(model)