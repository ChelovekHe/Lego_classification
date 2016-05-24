from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from Lego.extend import listdir_no_hidden
import keras


def initial_cnn_model(nb_class):
    # initial cnn model
    model = Sequential()

    # first convolution level, 4 kernels, each 5*5
    # activate function "relu"
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, 30, 30)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # flatten the feature maps from last level
    # dense is the hidden level, 16 feature maps from last level
    # 5*5 = (((30-5+1)-3+1)/2-3+1)/2
    # therefore 400 neurons, initialized as "normal"
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # softmax classification
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    # ----------------------------
    # start training
    # SGD + momentum
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

    return model


def train_model(model):
    data, label = load_data()
    test_data, test_label = load_test_data()

    label = np_utils.to_categorical(label)
    test_label = np_utils.to_categorical(test_label)

    batch_size = 100
    nb_epoch = 60

    check = keras.callbacks.ModelCheckpoint('../data/lego_identify_best.h5', monitor='val_loss',
                                            verbose=0, save_best_only=True, mode='auto')
    model.fit(data, label, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=1,
              validation_data=(test_data, test_label), callbacks=[check])
    return model


def load_data():
    path = '../info/'
    data = np.empty((1, 1, 30, 30), dtype='float32')
    label = []
    list1 = listdir_no_hidden(path)
    for i in range(0, len(list1)-1):
        list2 = listdir_no_hidden(path+list1[i])
        for j in range(0, len(list2)):
            img = Image.open(path+list1[i]+'/'+list2[j])
            arr = np.asarray(img, dtype='float32').reshape((1, 1, 30, 30))
            # arr = (arr - np.mean(arr)) / np.std(arr)
            data = np.append(data, arr, axis=0)
            label.append(i)
    label = np.asarray(label, dtype='uint8').reshape((len(label), ))
    data = data[1:, :, :, :]
    scale = np.max(data)
    data /= scale
    mean = np.std(data)
    data -= mean
    # data = (data - np.mean(data)) / np.std(data)
    return data, label


def load_test_data():
    path = '../info2/'
    data = np.empty((1, 1, 30, 30), dtype='float32')
    label = []
    list1 = listdir_no_hidden(path)
    for i in range(0, len(list1)):
        list2 = listdir_no_hidden(path+list1[i])
        for j in range(0, len(list2)):
            img = Image.open(path+list1[i]+'/'+list2[j])
            arr = np.asarray(img, dtype='float32').reshape((1, 1, 30, 30))
            # arr = (arr - np.mean(arr)) / np.std(arr)
            data = np.append(data, arr, axis=0)
            label.append(i)
    label = np.asarray(label, dtype='uint8').reshape((len(label), ))
    data = data[1:, :, :, :]
    data /= np.max(data)
    data -= np.std(data)
    # data = (data - np.mean(data)) / np.std(data)
    return data, label


if __name__ == '__main__':
    model = initial_cnn_model(5)
    # model.load_weights('../data/lego_identify.h5')
    model = train_model(model)
    model.save_weights('../data/lego_identify.h5', overwrite=True)
