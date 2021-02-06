# Facial Expression Analyzer (FEA)

## Data
### Datasets
Extract the datasets after a commit:
```tar -xvzf train/datasets.gzip```

Compress the datasets before a commit:
```tar -cz train/Datasets/ > train/datasets.gzip```

### Test pics
Extract the datasets after a commit:
```tar -xvzf test/test_pics.gzip```

Compress the datasets before a commit:
```tar -cz test/Test_pics/ > test/test_pics.gzip```

## Code
Python version: 3

### Basic requirements
`python -m pip install pipenv`
`pipenv install`

### Demo
Our CNN model is designed to take in a 48x48 image, so we test images of various sizes as
follows (steps on how a predic􀆟on is made in ‘predic􀆟on.py’):

## Single Images
Pass an image (ndarray) through display_expression(img, model, mode=0), a long
with a CNN model. Mode 0 means it's for a single image and will show a predic􀆟on vector along
with the image’s predic􀆟on. This func􀆟on uses a pretrained model ‘haarcascade’ to detect a
face and create a bounding box around it. The resul􀆟ng image is then extracted and rescaled to
48x48 and sent through our model. The resul􀆟ng predic􀆟on is displayed on the image.

## Webcam feed
We call the func􀆟on detect_emotions_webcam(model), which runs your webcam feed
and displays it to you if a face is detected. It calls d isplay_expression(img, model,
mode=1) b ut this 􀆟me in video mode, so it doesn’t constantly show a predic􀆟on vector. It
predicts expression on each frame in real-􀆟me due to haar cascades fast frontal face detec􀆟on,
which would be ideal for a twitch streamer.
