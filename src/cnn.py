import os
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from Lego.extend import listdir_no_hidden

def keras_cnn():
    data, label = load_data()
    label = np_utils.to_categorical(label)

    batch_size = 20
    nb_classes = label.shape[1]
    nb_epoch = 20

    # number of convolutional filters to use
    nb_filters1 = 32
    nb_filters2 = 64
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv1 = 3
    nb_conv2 = 3

    # initial cnn model
    model = Sequential()

    # first convolution level, 4 kernels, each 5*5
    # activate function "relu"
    model.add(Convolution2D(nb_filters1, nb_conv1, nb_conv1, border_mode='valid', input_shape=data.shape[-3:]))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(nb_filters2, nb_conv2, nb_conv2))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # flatten the feature maps from last level
    # dense is the hidden level, 16 feature maps from last level
    # 5*5 = (((30-5+1)-3+1)/2-3+1)/2
    # therefore 400 neurons, initialized as "normal"
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # softmax classification
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # ----------------------------
    # start training
    # SGD + momentum
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

    model.fit(data, label, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=1, validation_split=0.2)


def load_data():
    data = np.empty((1, 1, 30, 30), dtype='float32')
    label = []
    path = '../info/'
    list1 = listdir_no_hidden(path)
    for i in range(0, len(list1)-1):
        list2 = listdir_no_hidden(path+list1[i])
        for j in range(0, len(list2)):
            img = Image.open(path+list1[i]+'/'+list2[j])
            arr = np.asarray(img, dtype='float32').reshape((1, 1, 30, 30))
            normed = (arr - np.mean(arr)) / np.std(arr)
            data = np.append(data, arr, axis=0)
            label.append(i-1)
    label = np.asarray(label, dtype='uint8').reshape((len(label), ))
    data = data[1:, :, :, :]
    scale = np.max(data)
    data /= scale
    mean = np.std(data)
    data -= mean
    # data = (data - np.mean(data)) / np.std(data)
    return data, label


if __name__ == '__main__':
    keras_cnn()
