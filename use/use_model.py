import cv2
from out.predict import display_expression

def detect_emotions_image(path, model, output_file=None):
    full_img = cv2.imread(path)
    display_expression(full_img, model, output_file=output_file)

def detect_emotions_webcam(model, output_file=None):
    webcam = cv2.VideoCapture(0)

    while True:
        capture, frame = webcam.read()

        if capture == False:
            break

        # in video mode
        display_expression(frame, model, mode=1, output_file=output_file)