#!/usr/bin/env python
# coding: utf-8

# # Creating a Neural Network to detect High Latitude Dust events
#
# Airborne dust, including dust storms and weaker dust traces, can have deleterious and hazardous effects on human health, agriculture, solar power generation, and aviation. In this notebook, we will use MODIS RGB satellite product, which is labelled for dust events using Imagelabeler.
#
# The input to the model is obtained from worldview, and the events are annotated using the imagelabeler tool. This script is used to download images in a ML ready format necessesary for this script to work. For this demo, the input is already pre-formatted. the pre-formatted data is available to download form this link: s3://impact-datashare/hld-labeled

# ### Imports
# imports necessesary python packages
#
# If any package is missing from your environment, please create a empty python shell and use the following snippet to install missing packages
#
# !pip install package_name

# In[19]:

import argparse
import boto3
import cv2
import imgaug as ia
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os

import rasterio
import tensorflow as tf
import tensorflow.keras.backend as K

from PIL import Image
from glob import glob
from imgaug import augmenters as iaa

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    UpSampling2D,
    MaxPooling2D,
    Dropout,
    concatenate,
    Conv2DTranspose,
    BatchNormalization,
    Flatten
)

ia.seed(2)


# In[4]:


# configuration variables
input_size = 256
n_channels = 3
model_save_path = 'test_model.h5'


# ### Creating dataset generator and augmentations
# we use keras sequence library (https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence) to load a subset of input images in memory at a time and process in batches. This makes sure we are not running out of memory trying to load all images at once. imgaug library (https://github.com/aleju/imgaug) is used to augment images by translations and introducing random noise to inclrease the variability in input images and help with model generalization.

# In[29]:


class UnetGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data_path,
                 to_fit=True, batch_size=1, dim=(256, 256),
                 n_channels=3, shuffle=True, augment=True):
        'Initialization'
        self.data_path = data_path
        self.tif_list = []
        self.mask_list = []
        for filename in glob(f'{data_path}*.tif*'):
            self.tif_list.append(filename)

        self.to_fit = to_fit
        self.augment = augment
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.n = 0
        self.max = self.__len__()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.tif_list) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.tif_list))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        tif_list_temp = [self.tif_list[k] for k in indexes]

        # Generate data
        X = self._generate_X(tif_list_temp)

        #  preprocess and augment data

        if self.to_fit:
            y = self._generate_y(tif_list_temp)

            if self.augment:
                seq = make_augmentations()
                images_aug = list()
                labels_aug = list()
                for i in range(len(X)):
                    image, label = seq(
                        image=X[i].astype('float32'),
                        segmentation_maps=np.expand_dims(
                            y[i], 0).astype('uint8')
                    )
                    images_aug.append(image)
                    labels_aug.append(label[0, :, :, :])

                return np.array(images_aug), np.array(labels_aug)
            else:
                # import ipdb; ipdb.set_trace()
                return X, y[:,:,:,0]

        else:
            return X

    def _generate_X(self, tif_list_temp):
        'Generates data containing batch_size images'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(tif_list_temp):
            # Store sample
            X[i, ] = _load_tif_image(ID, self.dim)

        return X

    def _generate_y(self, tif_list_temp):
        'Generates data containing batch_size masks'
        y = np.empty((self.batch_size, *self.dim, 1), dtype='float32')

        # Generate data
        for i, ID in enumerate(tif_list_temp):
            # replace tif/tiff with the _bitmap.png or .bmp, depending on the dataset
            # to look for corresponding masks for the imput images
            y[i, ] = _load_grayscale_image(ID.replace(
                '.tiff', '_bitmap.png'), self.dim
            )

        return y

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result


def _load_grayscale_image(image_path, dim):
    'Load grayscale image'
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255

    return np.expand_dims(cv2.resize(img, dim), -1)


def _load_tif_image(image_path, dim):
    'load tif image'

    with rasterio.open(image_path, 'r') as data:
        return cv2.resize(
            np.moveaxis(data.read(), 0, -1), dim
        )


def sometimes(aug):
    return iaa.Sometimes(0.5, aug)

def make_augmentations():

    return iaa.Sequential([
        sometimes(iaa.CoarseDropout(0.1, size_percent=0.2)),
        sometimes(
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                # scale images to 80-120% of their size,
                # individually per axis
                translate_percent={
                    "x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                # translate by -20 to +20 percent (per axis)
                # rotate by -45 to +45 degrees
                rotate=(-10, 10),
                shear=(-5, 5),  # shear by -16 to +16 degrees
            ),
        ),
        sometimes(iaa.ElasticTransformation(alpha=10, sigma=1))
    ],
        random_order=True
    )


# ## Creating model
# the model we are using for this demo is Unet segmentation model (https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/). The architecture is a stack of convolutions followed by de-convolutions. This model assigns a class label to each pixel of the input and gives a output matching the size of the input. The resulting output, once trained with high-latitude dust(HLD) masks, will segment any given image to HLD and non-HLD pixels

# In[6]:


num_layers = 2
input_shape = (input_size, input_size, 3)
inputs = Input(input_shape)

def bn_conv_relu(input, filters, bachnorm_momentum, **conv2d_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2D(filters, **conv2d_args)(x)
    return x


def bn_upconv_relu(input, filters, bachnorm_momentum, **conv2d_trans_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2DTranspose(filters, **conv2d_trans_args)(x)
    return x

def build_model():
    filters = 24
    upconv_filters = 32

    kernel_size = (3, 3)
    activation = 'relu'
    strides = (1, 1)
    padding = 'same'
    kernel_initializer = 'he_normal'
    output_activation = 'sigmoid'

    conv2d_args = {
        'kernel_size': kernel_size,
        'activation': activation,
        'strides': strides,
        'padding': padding,
        'kernel_initializer': kernel_initializer
    }

    conv2d_trans_args = {
        'kernel_size': kernel_size,
        'activation': activation,
        'strides': (2, 2),
        'padding': padding,
    }

    bachnorm_momentum = 0.01

    pool_size = (2, 2)
    pool_strides = (2, 2)
    pool_padding = 'valid'

    maxpool2d_args = {
        'pool_size': pool_size,
        'strides': pool_strides,
        'padding': pool_padding,
    }

    x = Conv2D(filters, **conv2d_args)(inputs)
    c1 = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(c1, filters, bachnorm_momentum, **conv2d_args)
    x = MaxPooling2D(**maxpool2d_args)(x)

    down_layers = []

    for l in range(num_layers):
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        down_layers.append(x)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = MaxPooling2D(**maxpool2d_args)(x)

    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

    for conv in reversed(down_layers):
        x = concatenate([x, conv])
        x = bn_conv_relu(
            x, upconv_filters, bachnorm_momentum, **conv2d_args
        )
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_upconv_relu(
            x, filters, bachnorm_momentum, **conv2d_trans_args
        )

    x = concatenate([x, c1])
    x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)

    outputs = Conv2D(
        1,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=output_activation,
        padding='valid')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model


class PlotLearning(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()

        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()

        plt.show();


def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # input data and model directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--eval', type=str, default=os.environ.get('SM_CHANNEL_EVAL'))
    return parser.parse_known_args()


def assumed_role_session():
    # Assume the "notebookAccessRole" role we created using AWS CDK.
    client = boto3.client('sts')
    creds = client.assume_role(
        RoleArn=ROLE_ARN,
        RoleSessionName=ROLE_NAME
    )['Credentials']
    return boto3.session.Session(
        aws_access_key_id=creds['AccessKeyId'],
        aws_secret_access_key=creds['SecretAccessKey'],
        aws_session_token=creds['SessionToken'],
        region_name='us-east-1'
    )


def download_data(data, split):
    split_folder = f"data/{split}"
    if not(os.path.exists(split_folder)):
        os.makedirs(split_folder)
    session = assumed_role_session()
    s3_connection = session.resource('s3')
    splits = data.split('/')
    bucket_name = splits[2]
    bucket = s3_connection.Bucket(bucket_name)
    objects = list(bucket.objects.filter(Prefix="/".join(splits[3:])))
    for iter_object in objects:
        print(iter_object.key)
        splits = iter_object.key.split('/')
        if splits[-1]:
            filename = f"{split_folder}/{splits[-1]}"
            bucket.download_file(iter_object.key, filename)


if __name__ =='__main__':
    args, _ = parse_args()

    # ... load from args.train and args.test, train a model, write model to args.model_dir.

    # ### loading the dataset into generator
    # since we have two splits (train, val), we create two generators: one for train, and one for validation split.

    # prepare data from s3
#     download_data(args.train, 'train')
#     download_data(args.eval, 'eval')
#     download_data(args.test, 'test')
    print(args.train)
    train_generator = UnetGenerator(args.train, batch_size=4)
    val_generator = UnetGenerator(args.eval, batch_size=4)

    # ### training the model with the generators
    # 'steps_per_epoch' defines the number of times the generator should be called for each epoch. this number is the number of input samples (54) divided by the batch_size (4) ~= 13. similrly, validation_step is num of images in validation split (38) divided by batch_size ~= 10
    #
    # 'epochs' just need to be sufficiently large, since we are using EarlyStopping callback to preemptively stop model training if the loss does not improve for several epochs

    # ### Compiling the model with appropriate optimizer, loss and metric
    # For this model, we are using Adam (https://arxiv.org/abs/1412.6980) optimizer to train the model. since each pixel is either 0 (non-hld) or 1 (hld) pixel, we use binary crossentropy as the loss function. Accuracy is used to monitor the performance of the model.

    # ### Creating callbacks
    # Callbacks are pieces of code that get executed every time the model trains through one pass of all available input images. This is particularly useful functionality provided by keras to do various tasks such as stopping training if the model does not improve significantly, saving the weights of the best model only, and plotting training graph to better understand how the model learns in training phase.
    plot_learning = PlotLearning()
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20,
                      verbose=1, mode="auto"),
        ModelCheckpoint(filepath=args.model_dir,
                        verbose=1, save_best_only=True),
        plot_learning,
    ]

    model = build_model()

    model.compile(
        optimizer=Adam(
                    learning_rate=args.learning_rate,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-07,
                    amsgrad=False,
                ),
        loss='binary_crossentropy',
        metrics=["accuracy"]
    )

    model.fit_generator(
        train_generator,
        epochs=args.epochs,
        steps_per_epoch=13,
        validation_data=val_generator,
        callbacks=callbacks,
        validation_steps=10,
    )
