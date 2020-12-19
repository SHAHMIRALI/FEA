import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.utils import plot_model
from common.constants import TRAIN_DIR, VALIDATION_DIR, MODEL_PATH, IMG_DIM, BATCH_SIZE, EPOCHS, MODEL_PATH_test
from train.model import create_model

#Step 1 Augmenting data

augmentation_generator_train = ImageDataGenerator(horizontal_flip=True)
augmentation_generator_val = ImageDataGenerator(horizontal_flip=True)

x_train = augmentation_generator_train.flow_from_directory(TRAIN_DIR,target_size=(IMG_DIM,IMG_DIM),color_mode="grayscale",
                                                    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)

x_val = augmentation_generator_val.flow_from_directory(VALIDATION_DIR,target_size=(IMG_DIM,IMG_DIM),color_mode="grayscale",
                                                    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)


#Step 2 create CNN using Keras
model = create_model()
model.summary()

# Step 3: Training with validation
rlr_function = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001, mode='auto')

data = model.fit(x=x_train, steps_per_epoch=x_train.n//x_train.batch_size, epochs=EPOCHS, validation_data = x_val,
                 validation_steps = x_val.n//x_val.batch_size, callbacks=[rlr_function])

model.save(MODEL_PATH)

# SHOW GRAPHS
# Training vs Validation accuracy
plt.plot(data.history['accuracy'])
plt.plot(data.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Loss graph
plt.plot(data.history['loss'])
plt.plot(data.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

