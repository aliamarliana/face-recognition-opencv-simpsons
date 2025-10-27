# ======================================================================
# IMPORTS
# ======================================================================
import os
import caer       # Lightweight Computer Vision library (used for data handling)
import canaro     # Deep Learning training utilities built on top of Keras
import numpy as np
import cv2 as cv
import gc
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers.legacy import SGD

# ======================================================================
# DATASET SETUP
# ----------------------------------------------------------------------
# We use the Simpsons Character Dataset to train a face recognition model.
# Images are converted to grayscale and resized for faster training.
# ======================================================================
IMG_SIZE = (80, 80)
channels = 1
char_path = r'../input/the-simpsons-characters-dataset/simpsons_dataset'

# ======================================================================
# EXPLORING DATA
# ----------------------------------------------------------------------
# Create a dictionary mapping each character to the number of images.
# Then sort by descending frequency to focus on the most represented characters.
# ======================================================================
char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))

char_dict = caer.sort_dict(char_dict, descending=True)
char_dict

# ======================================================================
# SELECTING TOP CHARACTERS
# ----------------------------------------------------------------------
# We focus only on the 10 characters with the most training samples
# to improve model balance and training efficiency.
# ======================================================================
characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break
characters

# ======================================================================
# DATA PREPARATION
# ----------------------------------------------------------------------
# Preprocess images from selected character folders.
# caer handles loading, resizing, and grayscale conversion.
# ======================================================================
train = caer.preprocess_from_dir(
    char_path, 
    characters, 
    channels=channels, 
    IMG_SIZE=IMG_SIZE, 
    isShuffle=True
)

# Check number of training samples
len(train)

# ======================================================================
# DATA VISUALIZATION
# ----------------------------------------------------------------------
# Display one sample image to verify preprocessing.
# (Note: OpenCV images use BGR order, but matplotlib expects RGB/gray.)
# ======================================================================
plt.figure(figsize=(30,30))
plt.imshow(train[0][0], cmap='gray')
plt.show()

# ======================================================================
# FEATURE AND LABEL EXTRACTION
# ----------------------------------------------------------------------
# Split the data into feature arrays (X) and labels (y).
# Normalize pixel values and one-hot encode labels for classification.
# ======================================================================
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)
featureSet = caer.normalize(featureSet)
labels = to_categorical(labels, len(characters))

# ======================================================================
# TRAIN-VALIDATION SPLIT
# ----------------------------------------------------------------------
# 80% training data, 20% validation data.
# This ensures fair evaluation during training.
# ======================================================================
x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=.2)

# Clean up unused variables to optimize memory usage
del train, featureSet, labels
gc.collect()

# ======================================================================
# TRAINING PARAMETERS
# ----------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 10

# ======================================================================
# DATA AUGMENTATION
# ----------------------------------------------------------------------
# Introduce controlled randomness to the data (rotation, zoom, etc.)
# to make the model more robust and reduce overfitting.
# ======================================================================
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# ======================================================================
# MODEL ARCHITECTURE
# ----------------------------------------------------------------------
# A CNN (Convolutional Neural Network) inspired by VGG structure.
# - Multiple Conv2D layers extract hierarchical features.
# - Dropout layers prevent overfitting.
# - Softmax output layer performs multi-class classification.
# ======================================================================
output_dim = 10
w, h = IMG_SIZE[:2]

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(w, h, channels)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(256, (3, 3), padding='same', activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Flatten(),
    Dropout(0.5),
    Dense(1024, activation='relu'),
    Dense(output_dim, activation='softmax')   # Output layer
])

model.summary()

# ======================================================================
# MODEL COMPILATION AND TRAINING
# ----------------------------------------------------------------------
# Using Stochastic Gradient Descent (SGD) with Nesterov momentum:
# - Stable convergence
# - Prevents overshooting minima
# Learning rate scheduling is handled via Canaro’s utility.
# ======================================================================
optimizer = SGD(learning_rate=0.001, decay=1e-7, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

training = model.fit(
    train_gen,
    steps_per_epoch=len(x_train)//BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    validation_steps=len(y_val)//BATCH_SIZE,
    callbacks=callbacks_list
)

print(characters)

# ======================================================================
# TESTING PHASE
# ----------------------------------------------------------------------
# Load an unseen image, preprocess it, and make a prediction.
# The model outputs the most probable character.
# ======================================================================
test_path = r'../input/the-simpsons-characters-dataset/kaggle_simpson_testset/kaggle_simpson_testset/charles_montgomery_burns_0.jpg'
img = cv.imread(test_path)

plt.imshow(img)
plt.show()

def prepare(image):
    """Convert image to grayscale, resize, and reshape for model input."""
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, IMG_SIZE)
    image = caer.reshape(image, IMG_SIZE, 1)
    return image

predictions = model.predict(prepare(img))

# Display the predicted character
print(characters[np.argmax(predictions[0])])
