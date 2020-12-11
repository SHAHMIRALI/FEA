from keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from keras.layers import BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from constants import TRAIN_DIR, TEST_DIR, MODEL_PATH, IMG_DIM, BATCH_SIZE, EPOCHS


def create_model(lr=0.0005):
    # Initialising the CNN
    model = Sequential()

    # First CNN layer
    # 64 filters 3x3, images are 48x48 and 1 channel cuz grayscale
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(IMG_DIM, IMG_DIM, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Divide height and width of conv block by 2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # can mess around with this hyperparameter (drops random data)
    model.add(Dropout(0.2))

    # Second CNN layer
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Third CNN layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Fourth CNN layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Flattening
    model.add(Flatten())

    # Fully connected layer 1st layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Fully connected layer 2nd layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # 7 labels
    model.add(Dense(7, activation='softmax'))

    opt = Adam(lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model