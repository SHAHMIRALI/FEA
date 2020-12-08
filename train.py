import os
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.utils import plot_model
from constants import TRAIN_DIR, VALIDATION_DIR, MODEL_PATH, IMG_DIM, BATCH_SIZE, EPOCHS
from model import create_model


from IPython.display import SVG, Image
from livelossplot import PlotLossesKeras
import tensorflow as tf

#Step 1 Augmenting data

datagen_train = ImageDataGenerator(horizontal_flip=True)

train_generator = datagen_train.flow_from_directory(TRAIN_DIR,
                                                    target_size=(IMG_DIM,IMG_DIM),
                                                    color_mode="grayscale",
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=True)

datagen_val = ImageDataGenerator(horizontal_flip=True)

validation_generator = datagen_val.flow_from_directory(VALIDATION_DIR,
                                                    target_size=(IMG_DIM,IMG_DIM),
                                                    color_mode="grayscale",
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=False)


#Step 2 create CNN using Keras
model = create_model()
model.summary()

# Step 3: Training with validation
steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.00001, mode='auto')
# checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy',
#                              save_weights_only=True, mode='max', verbose=1)
callbacks = [reduce_lr]

history = model.fit(
    x=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data = validation_generator,
    validation_steps = validation_steps,
    #uncomment to see graphs
    callbacks=callbacks
)

# Training vs Validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save(MODEL_PATH)