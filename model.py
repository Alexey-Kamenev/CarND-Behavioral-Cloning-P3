import argparse
import csv
import h5py
import numpy as np
import os
import sys
from PIL import Image
from PIL import ImageEnhance as ime

from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

image_dims = [160, 320, 3]

def train_generator(data_dir, samples, batch_size=64):
    def get_factor(radius):
        return np.random.uniform(1. - radius, 1. + radius)

    num_samples = len(samples)

    # In the case of the simulator, such (and similar) distortions add only marginal value
    # in preventing overfitting so not using them.
    enhancers = [
        [ime.Color,      0.4],
        [ime.Contrast,   0.3],
        [ime.Brightness, 0.3],
        [ime.Sharpness,  0.3],
    ]

    while True:
        np.random.shuffle(samples)
        for start in range(0, num_samples, batch_size):
            batch_samples = samples[start:start + batch_size]
            images  = np.array([]).reshape([0] + image_dims)
            targets = []
            np.random.shuffle(enhancers)
            for sample in batch_samples:
                img_idx     = np.random.randint(0, 3)
                corrections = [0, 0.2, -0.2]
                img = Image.open(os.path.join(data_dir, sample[img_idx].strip()))
                # for cur_e in enhancers:
                #     e   = cur_e[0](img)
                #     img = e.enhance(get_factor(cur_e[1]))
                img = np.array(img.getdata()).reshape([1] + image_dims)
                tgt = float(sample[3]) + corrections[img_idx]
                if np.random.binomial(1, 0.5) == 1:
                    img  = np.fliplr(img)
                    tgt = -tgt
                images = np.vstack((images, img))
                targets.append(tgt)
            yield (images, np.array(targets))

def val_generator(data_dir, samples, batch_size=64):
    num_samples = len(samples)
    while True:
        for start in range(0, num_samples, batch_size):
            batch_samples = samples[start:start + batch_size]
            images  = np.array([]).reshape([0] + image_dims)
            targets = []
            for sample in batch_samples:
                img = Image.open(os.path.join(data_dir, sample[0].strip()))
                img = np.array(img.getdata()).reshape([1] + image_dims)
                tgt = float(sample[3])
                images = np.vstack((images, img))
                targets.append(tgt)
            yield (images, np.array(targets))
                
def create_train_val_gen(base_data_dir, data_dirs, batch_size=64, skip_header=False):
    lines = []
    for d in data_dirs:
        base_dir = os.path.join(data_dir, d)
        with open(os.path.join(base_dir, 'driving_log.csv')) as f:
            cur_lines = csv.reader(f)
            if skip_header:
                next(cur_lines)
            for line in cur_lines:
                center_file = os.path.join(base_dir, line[0].strip())
                left_file   = os.path.join(base_dir, line[1].strip())
                right_file  = os.path.join(base_dir, line[2].strip())
                if os.path.exists(center_file) and os.path.exists(left_file) and os.path.exists(right_file):
                    line[0] = os.path.join(d, line[0].strip())
                    line[1] = os.path.join(d, line[1].strip())
                    line[2] = os.path.join(d, line[2].strip())
                    lines.append(line)
    # Create train/val split.
    train_samples, val_samples = train_test_split(lines, test_size=0.05)
    if len(train_samples) % batch_size != 0:
        train_samples = train_samples[:-(len(train_samples) % batch_size)]
    if len(val_samples) % batch_size != 0:
        val_samples = val_samples[:-(len(val_samples) % batch_size)]
    train_gen = train_generator(data_dir, train_samples, batch_size=batch_size)
    val_gen   = val_generator(data_dir, val_samples, batch_size=batch_size)
    return train_gen, val_gen, len(train_samples), len(val_samples)

# Neural net architectures.

# Simple net for testing overall pipeline.
def net_simple(model):
    model.add(Convolution2D(16, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

# Simple convolutional net with batch norm.
def net_simple_bn(model):
    model.add(Convolution2D(32, 5, 5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, 5, 5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(128, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization(mode=1))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(BatchNormalization(mode=1))
    model.add(Activation('relu'))

# A bit larger net with no FC layers (global pooling instead).
def net_medium_bn(model):
    model.add(Convolution2D(64, 5, 5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(128, 5, 5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(128, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('--checkpoint', help='Path to checkpoint file.')
    args = parser.parse_args()

    batch_size = 64
    num_epochs = 10
    data_dir = '../data/p3/ak'
    #dirs     = ['track1_kb', 'track1_mouse']
    dirs     = ['track1_kb', 'track1_mouse', 'track2_mouse_center', 'track2_mouse_center_02']
    #dirs     = ['one_lap']
    train_gen, val_gen, train_size, val_size = create_train_val_gen(data_dir, dirs, batch_size=batch_size)
    #data_dir = '../data/p3/data'
    #train_gen, val_gen, train_size, val_size = create_train_val_gen(data_dir, batch_size=batch_size, skip_header=True)
    print('Train size: {}'.format(train_size))
    print('Val size  : {}'.format(val_size))
    
    if not args.checkpoint:
        model = Sequential()
        model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=image_dims))
        model.add(Lambda(lambda x: x/127.5 - 1.0))
        #net_01(model)
        net_simple_bn(model)
        #net_medium_bn(model)
        model.add(Dense(1))

        #optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        optimizer = optimizers.adam()
        model.compile(loss='mean_squared_error', optimizer=optimizer)
    else:
        print('Starting from checkpoint: {}'.format(args.checkpoint))
        model = load_model(args.checkpoint)
    checkpointer = ModelCheckpoint(filepath="./weights_{epoch:02d}.h5")
    model.fit_generator(train_gen, samples_per_epoch=train_size,
                        validation_data=val_gen, nb_val_samples=val_size,
                        nb_epoch=num_epochs, nb_worker=4, pickle_safe=True,
                        callbacks=[checkpointer])

    #model.save('model.h5')
