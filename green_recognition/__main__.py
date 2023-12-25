import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow import random
import random as rn
import cv2
import numpy as np
from tqdm import tqdm
import os
from random import shuffle
from zipfile import ZipFile
from PIL import Image

X = []
Z = []
IMG_SIZE = 150
BLACK_CABBAGE_DIR = '../green_recognition/data/greens/black_cabbage'
CHARD_DIR = '../data/greens/chard'
CRESS_DIR = '../data/greens/cress'
CURLY_LETTUCE_DIR = '../data/greens/curly_lettuce'
DILL_DIR = '../data/greens/dill'
GREEN_GARLIC_DIR = '../data/greens/green_garlic'
ICEBERG_DIR = '../data/greens/iceberg'
LEEK_DIR = '../data/greens/leek'
LETTUCE_DIR = '../data/greens/lettuce'
MINT_DIR = '../data/greens/mint'
NETTLE_DIR = '../data/greens/nettle'
PARSLEY_DIR = '../data/greens/parsley'
PORTULACA_DIR = '../data/greens/portulaca'
ROCKET_DIR = '../data/greens/roquette'
SCALLION_DIR = '../data/greens/scallion'
SPINACH_DIR = '../data/greens/spinach'


def assign_label(img, green_type):
    return green_type


def make_train_data(green_type, DIR):
    for img in tqdm(os.listdir(DIR)):
        label = assign_label(img, green_type)
        path = os.path.join(DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(np.array(img))
            Z.append(str(label))


if __name__ == "__main__":
    make_train_data('Black Cabbage', BLACK_CABBAGE_DIR)
    make_train_data('Chard', CHARD_DIR)

    print(len(X))
    le = LabelEncoder()
    Y = le.fit_transform(Z)
    Y = to_categorical(Y, 5)
    X = np.array(X)
    X = X / 255
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    np.random.seed(42)
    rn.seed(42)
    random.set_seed(42)
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(5, activation="softmax"))
    batch_size = 128
    epochs = 50

    from keras.callbacks import ReduceLROnPlateau

    red_lr = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.1)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    History = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                  epochs=epochs, validation_data=(x_test, y_test),
                                  verbose=1, steps_per_epoch=x_train.shape[0] // batch_size)

    pred = model.predict(x_test)
    pred_digits = np.argmax(pred, axis=1)

    i = 0
    prop_class = []
    mis_class = []

    for i in range(len(y_test)):
        if (np.argmax(y_test[i]) == pred_digits[i]):
            prop_class.append(i)
        if (len(prop_class) == 8):
            break

    i = 0
    for i in range(len(y_test)):
        if (not np.argmax(y_test[i]) == pred_digits[i]):
            mis_class.append(i)
        if (len(mis_class) == 8):
            break
