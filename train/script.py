import sys
import numpy as np
from PIL import Image as img

filename = "./Datasets/train.csv"

j = 0

#40-40-20 split for train/test/validation
with open(filename, 'r') as f:
    print(f.readline())
    for i, line in enumerate(f):
        emotion, pixels = line.split(",")
        pixels = pixels[1:-2]
        pixels = pixels.split(" ")
        ar = np.asarray(pixels, dtype=np.uint8)
        ar = ar.reshape((48, 48))
        pic = img.fromarray(ar, "L")

        if emotion == "0":
            if j % 5 == 0:
                pic.save(r"./Datasets/Kaggle/validation/Angry/{}.jpg".format(i), "JPEG")
            elif j % 2 == 0:
                pic.save(r"./Datasets/Kaggle/test/Angry/{}.jpg".format(i), "JPEG")
            else:
                pic.save(r"./Datasets/Kaggle/train/Angry/{}.jpg".format(i), "JPEG")
        elif emotion == "1":
            if j % 5 == 0:
                pic.save(r"./Datasets/Kaggle/validation/Disgust/{}.jpg".format(i), "JPEG")
            elif j % 2 == 0:
                pic.save(r"./Datasets/Kaggle/test/Disgust/{}.jpg".format(i), "JPEG")
            else:
                pic.save(r"./Datasets/Kaggle/train/Disgust/{}.jpg".format(i), "JPEG")
        elif emotion == "2":
            if j % 5 == 0:
                pic.save(r"./Datasets/Kaggle/validation/Fear/{}.jpg".format(i), "JPEG")
            elif j % 2 == 0:
                pic.save(r"./Datasets/Kaggle/test/Fear/{}.jpg".format(i), "JPEG")
            else:
                pic.save(r"./Datasets/Kaggle/train/Fear/{}.jpg".format(i), "JPEG")
        elif emotion == "3":
            if j % 5 == 0:
                pic.save(r"./Datasets/Kaggle/validation/Happy/{}.jpg".format(i), "JPEG")
            elif j % 2 == 0:
                pic.save(r"./Datasets/Kaggle/test/Happy/{}.jpg".format(i), "JPEG")
            else:
                pic.save(r"./Datasets/Kaggle/train/Happy/{}.jpg".format(i), "JPEG")
        elif emotion == "4":
            if j % 5 == 0:
                pic.save(r"./Datasets/Kaggle/validation/Sad/{}.jpg".format(i), "JPEG")
            elif j % 2 == 0:
                pic.save(r"./Datasets/Kaggle/test/Sad/{}.jpg".format(i), "JPEG")
            else:
                pic.save(r"./Datasets/Kaggle/train/Sad/{}.jpg".format(i), "JPEG")
        elif emotion == "5":
            if j % 5 == 0:
                pic.save(r"./Datasets/Kaggle/validation/Surprise/{}.jpg".format(i), "JPEG")
            elif j % 2 == 0:
                pic.save(r"./Datasets/Kaggle/test/Surprise/{}.jpg".format(i), "JPEG")
            else:
                pic.save(r"./Datasets/Kaggle/train/Surprise/{}.jpg".format(i), "JPEG")
        elif emotion == "6":
            if j % 5 == 0:
                pic.save(r"./Datasets/Kaggle/validation/Neutral/{}.jpg".format(i), "JPEG")
            elif j % 2 == 0:
                pic.save(r"./Datasets/Kaggle/test/Neutral/{}.jpg".format(i), "JPEG")
            else:
                pic.save(r"./Datasets/Kaggle/train/Neutral/{}.jpg".format(i), "JPEG")

        j += 1
