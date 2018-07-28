'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Concatenate, Reshape, Conv2D, MaxPool2D, Flatten, BatchNormalization, Deconv2D
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model, to_categorical
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    # z_mean, _, _ = encoder.predict([x_test, y_test],
    #                                batch_size=batch_size)
    # print(z_mean[:10])
    # plt.figure(figsize=(12, 10))
    # colors = np.argmax(y_test)
    # plt.scatter(z_mean[:, 0], z_mean[:, 1], c=colors.tolist())
    # plt.colorbar()
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    # plt.savefig(filename)
    # plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    label = np.zeros((47,))
    label[28] = 1
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([xi, xi, yi, yi])
            decoder_in = np.hstack((z_sample, label))
            x_decoded = decoder.predict(np.array([decoder_in]))
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
import csv
def load_dataset(fpath, mpath='emnist/emnist-balanced-mapping.txt'):
    X = []
    Y = []

    M = {}
    with open(fpath) as csv_file:
        reader = csv.reader(csv_file)
        for idx, row in enumerate(reader):
            Y.append(int(row[0]))
            X.append(np.transpose(np.reshape(np.array(row[1:], np.float32), (28, 28)), [1, 0]))
    X = np.array(X)
    Y = np.array(Y)

    with open(mpath) as map_file:
        for line in map_file:
            c_and_ascii = line.split()
            M[int(c_and_ascii[0].strip())] = str(chr(int(c_and_ascii[1].strip())))
    return X, to_categorical(Y), M

x_train, y_train, _ = load_dataset('emnist/emnist-balanced-train.csv')
x_test, y_test, _ = load_dataset('emnist/emnist-balanced-test.csv')


image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters

# intermediate_dim = 512
batch_size = 128
latent_dim = 4
epochs = 50
n_classes = y_train.shape[1]

# VAE model = encoder + decoder
# build encoder model
def get_cvae(input_shape, img_shape):
    X = Input(shape=input_shape, name='encoder_input')
    label = Input(shape=(n_classes, ))
    inputs = Concatenate()([X, label])

    x = Dense(input_shape[0], activation='relu')(inputs)
    x = Reshape(img_shape)(x)

    # Block 1
    x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)

    # Block 2
    x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)

    # Block 3
    x = Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)

    # Network in network
    x = Conv2D(64, kernel_size=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Latent space
    x = Flatten()(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    zc = Concatenate()([z, label])

    # instantiate encoder model
    encoder = Model([X, label], [z_mean, z_log_var, zc], name='encoder')
    encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim+n_classes,), name='z_sampling')
    x = Dense(16, activation='relu')(latent_inputs)
    x = BatchNormalization()(x)
    x = Reshape((4, 4, 1))(x)

    # Block 1
    x = Deconv2D(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Block 2
    x = Deconv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=3, strides=1, activation='relu')(x)
    x = BatchNormalization()(x)

    # Block 3
    x = Deconv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(x)
    outputs = Flatten()(x)
    # outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder([X, label])[2])
    vae = Model([X, label], outputs, name='vae_mlp')

    # vae loss
    reconstruction_loss = binary_crossentropy(X, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    return vae, encoder, decoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()

    vae, encoder, decoder = get_cvae(input_shape=(784,), img_shape=(28, 28, 1))
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    # if args.mse:
    #     reconstruction_loss = mse(X, outputs)
    # else:

    if args.weights:
        vae = vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit([x_train, y_train],
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([x_test, y_test], None))
        vae.save_weights('vae_mlp_mnist.h5')

    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae_mlp")