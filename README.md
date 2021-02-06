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

## Demo
Our CNN model is designed to take in a 48x48 image, so we test images of various sizes as
follows (steps on how a predic􀆟on is made in ‘predic􀆟on.py’):

### Single Images
Pass an image (ndarray) through display_expression(img, model, mode=0), a long
with a CNN model. Mode 0 means it's for a single image and will show a prediction vector along
with the image’s prediction. This function uses a pretrained model ‘haarcascade’ to detect a
face and create a bounding box around it. The resulting image is then extracted and rescaled to
48x48 and sent through our model. The resulting prediction is displayed on the image.

![image](https://user-images.githubusercontent.com/32078797/107105382-fcbeab00-67f3-11eb-820d-78c780a72480.png)
![image](https://user-images.githubusercontent.com/32078797/107105389-09db9a00-67f4-11eb-9139-9c5a526cc58c.png)

![image](https://user-images.githubusercontent.com/32078797/107105400-152ec580-67f4-11eb-88ac-286185a8a0b7.png)
![image](https://user-images.githubusercontent.com/32078797/107105406-1fe95a80-67f4-11eb-93ea-e2f0373ef648.png)


### Webcam feed
We call the function detect_emotions_webcam(model), which runs your webcam feed
and displays it to you if a face is detected. It calls display_expression(img, model,
mode=1) but this time in video mode, so it doesn’t constantly show a prediction vector. It
predicts expression on each frame in real-time due to haar cascades fast frontal face detection,
which would be ideal for a twitch streamer.

![image](https://user-images.githubusercontent.com/32078797/107105584-21675280-67f5-11eb-8426-84d76ff9a3d1.png)
![image](https://user-images.githubusercontent.com/32078797/107105432-3ee7ec80-67f4-11eb-8c4c-fce1e1e81615.png)

## Results
Test accuracies for our CNN and SVM models:

### CNN
![image](https://user-images.githubusercontent.com/32078797/107105604-3fcd4e00-67f5-11eb-8dbf-5c82f633f536.png)
![image](https://user-images.githubusercontent.com/32078797/107105609-4f4c9700-67f5-11eb-97ca-87c1aa3d8fc6.png)

### SVM
![image](https://user-images.githubusercontent.com/32078797/107105686-86bb4380-67f5-11eb-9535-eea8d8eda7da.png)
![image](https://user-images.githubusercontent.com/32078797/107105691-8f137e80-67f5-11eb-9842-f4d1fd0face0.png)

