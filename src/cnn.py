from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
# import matplotlib.pyplot as plt
import os
import keras
# import pylab as pl
import theano
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.cm as cm
import numpy.ma as ma


def listdir_no_hidden(path):
    list1 = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            list1.append(f)
    return list1

def initial_cnn_model(nb_class):
    # initial cnn model
    model = Sequential()

    # first convolution level, 4 kernels, each 5*5
    # activate function "relu"
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, 30, 30)))
    model.add(Activation('relu'))
    # model.add(Convolution2D(32, 3, 3))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    # model.add(Convolution2D(32, 3, 3))
    # model.add(Activation('relu'))
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
    nb_epoch = 80

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


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)
    return im


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                           dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
        col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

def visualise(model):
    # define theano funtion to get output of  first Conv layer
    get_featuremap = theano.function([model.layers[0].input], model.layers[0].get_output_at(0),
                                     allow_input_downcast=False)
    data, label = load_data()

    # num_fmap = 32  # number of feature map
    # for i in range(num_fmap):
    #     featuremap = get_featuremap(data[0:1])
    #     plt.imshow(featuremap[0][i], cmap=cm.Greys_r)  # visualize the first image's 4 feature map
    #     plt.show()
    #
    # C1 = get_featuremap(data[0:1])
    # C1 = np.squeeze(C1)
    # print("C1 shape : ", C1.shape)
    # W = np.squeeze(C1)
    # pl.figure(figsize=(15, 15))
    # pl.title('conv1 weights')
    # nice_imshow(pl.gca(), make_mosaic(W, 6, 6), cmap=cm.binary)


if __name__ == '__main__':
    model = initial_cnn_model(5)
    # model.load_weights('../data/lego_identify.h5')
    # visualise(model)

    # W = model.layers[0].W.get_value(borrow=True)
    # W = np.squeeze(W)
    # print("W shape : ", W.shape)




    model = train_model(model)
    model.save_weights('../data/lego_identify.h5', overwrite=True)