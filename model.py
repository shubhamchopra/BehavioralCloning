import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D
from keras.models import model_from_json
from keras.optimizers import Adam

from scipy.misc import imread, imsave

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from pandas import read_csv

import tensorflow as tf
import numpy as np

import random

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_dir', '', "Training data directory")
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")
flags.DEFINE_boolean('use_mirrored_data', False, "Supplement training data with mirrored data")

# log file name in the data directory
drive_log_file = 'driving_log.csv'

def get_image_path(data_path, filepath):
    '''
    We normalize the filepath here. This is done so we can use both absolute and relative paths
    easily.
    '''
    basename = os.path.basename(filepath)
    img_dir = os.path.join(data_path, 'IMG')
    return os.path.join(img_dir, basename)

def get_drive_data():
    '''
    We read the data log file and extract the location of images and the corresponding steering angles.
    We split this dataset of image names and steering angles into a training and test split to 
    generate validation data. We don't read the images just yet, since we usee a generator, and want to
    save on memory usage.
    '''
    data_path = FLAGS.training_dir
    logfile = os.path.join(data_path, drive_log_file)
    df = read_csv(logfile)[['center', 'steering']]
    df['center'] = df['center'].apply(lambda x: get_image_path(data_path, x))
    dfm = df.as_matrix()
    return train_test_split(dfm[:, 0], dfm[:, 1], test_size = 0.1)

def get_gen_data(Xin, yin):
    '''
    We build a data generator here, where each element is a batch of data, with the specified batch size.
    If the `use_mirrored_data` flag is set, we randomly choose to mirror the batch of data. This helps in
    supplementing the data set and reduces overfitting.
    '''
    while 1:
        Xs, ys = shuffle(Xin, yin)
        for offset in range(0, len(Xin), FLAGS.batch_size):
            end = offset + FLAGS.batch_size
            Xb = np.asarray([imread(im, mode='RGB') for im in Xs[offset:end]])
            yb = ys[offset:end]
            if FLAGS.use_mirrored_data and random.choice([True, False]):
#                imsave('origImage.jpg', Xb[0])
#                imsave('mirror.jpg', Xb[:,:,::-1][0])
                Xb = Xb[:,:,::-1]
                yb = -1 * yb
            yield Xb, yb

def get_model():
    row, col, ch = 160, 320, 3  # camera format
    print('generating comma.ai model')
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - .5, input_shape=(row, col, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    return model

def get_nvidia_model():
    row, col, ch = 160, 320, 3  # camera format
    print('generating nVidia model')
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0, input_shape=(row, col, ch)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(Activation('relu'))
    model.add(Dense(1164))
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(100))
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(50))
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(10))
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(1))
    return model

if __name__ == '__main__':
    Xtrain, Xval, ytrain, yval = get_drive_data()
    train_gen = get_gen_data(Xtrain, ytrain)
    val_gen = get_gen_data(Xval, yval)
    model = None
    optimizer = None
    if os.path.isfile('model.json'):
# we have a pre-estimated model, so we use it
        print('loading model...')
        with open('model.json', 'r') as f:
            js = f.readline()
            model = model_from_json(js)
        model.load_weights('model.h5')
# we use a lower learning rate, as this is a `tuning` run.
        optimizer = Adam(lr = 0.0001)
    else:
        model = get_nvidia_model()
        optimizer = Adam() 

    model.compile(optimizer = optimizer, loss = 'mse')
    if FLAGS.use_mirrored_data:
        print('Using mirrored data.')
    model.fit_generator(train_gen, samples_per_epoch = len(Xtrain), nb_epoch = FLAGS.epochs, validation_data = val_gen, nb_val_samples=len(Xval))

    print('writing model to file...')
    with open('model.json', 'w') as f:
        js = model.to_json()
        f.write(js)
    model.save_weights('model.h5')
    print('Done.')

