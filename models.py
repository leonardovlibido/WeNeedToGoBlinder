from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Dense, Flatten, Input, ELU
from keras.layers import Deconv2D, Reshape, Concatenate, Lambda, LeakyReLU
from keras.models import Model
import data_utils
from keras.models import load_model
from keras.callbacks import Callback
import os
import numpy as np
from keras import backend as K
from keras.layers.merge import concatenate as concat

def get_classification_model(n_class, input_shape=(28, 28, 1), print_summary=False):
    x = Input(shape=input_shape)

    y_hat = Conv2D(filters=32, kernel_size=3, padding='same', name='conv1_1')(x)
    y_hat = ELU()(y_hat)
    y_hat = Conv2D(filters=32, kernel_size=3, padding='same', name='conv1_2')(y_hat)
    y_hat = ELU()(y_hat)
    y_hat = BatchNormalization(name='bn1')(y_hat)
    y_hat = MaxPool2D(pool_size=2, name='max_pool1')(y_hat)
    y_hat = Dropout(0.2, name='dropout1')(y_hat)

    y_hat = Conv2D(filters=64, kernel_size=3, padding='same', name='conv2_1')(y_hat)
    y_hat = ELU()(y_hat)
    y_hat = Conv2D(filters=64, kernel_size=3, padding='same', name='conv2_2')(y_hat)
    y_hat = ELU()(y_hat)
    y_hat = BatchNormalization(name='bn2')(y_hat)
    y_hat = MaxPool2D(pool_size=2, name='max_pool2')(y_hat)
    y_hat = Dropout(0.2, name='dropout2')(y_hat)

    y_hat = Conv2D(filters=128, kernel_size=3, padding='same', name='conv3_1')(y_hat)
    y_hat = ELU()(y_hat)
    y_hat = Conv2D(filters=128, kernel_size=3, padding='same', name='conv3_2')(y_hat)
    y_hat = ELU()(y_hat)
    y_hat = BatchNormalization(name='bn3')(y_hat)
    y_hat = MaxPool2D(pool_size=2, name='max_pool3')(y_hat)
    y_hat = Dropout(0.2, name='dropout3')(y_hat)

    y_hat = Flatten(name='flatten')(y_hat)
    y_hat = Dense(512, name='fc1')(y_hat)
    y_hat = ELU()(y_hat)
    y_hat = Dropout(0.3, name='dropout4')(y_hat)
    y_hat = Dense(256, name='fc2')(y_hat)
    y_hat = ELU()(y_hat)
    y_hat = Dropout(0.3, name='dropout5')(y_hat)
    y_hat = Dense(32, name='fc3')(y_hat)
    y_hat = ELU()(y_hat)
    y_hat = Dropout(0.3, name='dropout6')(y_hat)
    y_hat = Dense(n_class, activation='softmax', name='softmax')(y_hat)

    model = Model(x, y_hat)
    if print_summary:
        model.summary()
    return model


def get_CVAE_model(feature_vector_generator, print_summary=False):
    encoder, input_img, final_encoded, z_mean, z_log_sigma = _get_CVAE_encoder(feature_vector_generator)
    decoder = _get_CVAE_decoder()

    model = Model(input_img, decoder(final_encoded))

    if print_summary:
        feature_vector_generator.summary()
        model.summary()
        encoder.summary()
        decoder.summary()
    return model, encoder, decoder, z_mean, z_log_sigma


def _get_CVAE_encoder(feature_vector_generator, input_shape=(28, 28, 1)):
    input_img = Input(shape=input_shape, name='autoencoder_input')

    flat_img = Flatten(name='flattenAutoEncoder')(input_img)
    c = feature_vector_generator(input_img)
    x = concat([flat_img, c])
    x = Dense(784)(x)
    x = Reshape(target_shape=(28, 28, 1))(x)

    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(32, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(16, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(16, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(8, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(8, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)
    x = BatchNormalization()(x)
    # 2x2x8
    encoder = Flatten(name='encoding_layer')(x)

    latent_dim = 64
    z_mean = Dense(latent_dim)(encoder)
    z_log_sigma = Dense(latent_dim)(encoder)

    z = Lambda(sampling, output_shape=(latent_dim,), name="sampling_add")([z_mean, z_log_sigma])

    final_encoded = concat([z, c])
    return Model(input_img, final_encoded), input_img, final_encoded, z_mean, z_log_sigma

def _get_CVAE_decoder(input_shape=(96,)):
    input_code = Input(input_shape, name='decoder_input')
    x = Dense(32, name='sampling_dense')(input_code)

    x = Reshape(target_shape=(2, 2, 8))(x)

    x = Deconv2D(8, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(8, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = BatchNormalization()(x)

    x = Deconv2D(16, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(16, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = BatchNormalization()(x)

    x = Deconv2D(32, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(32, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = BatchNormalization()(x)

    x = Deconv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = BatchNormalization()(x)

    decoded = Conv2D(1, kernel_size=5, strides=1, padding='valid', activation='tanh', name='decoding_layer')(x)
    return Model(input_code, decoded)


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def _get_encoder(input_shape=(28, 28, 1)):
    input_img = Input(shape=input_shape, name='encoder_input')

    x = Conv2D(128, kernel_size=3, strides=1, padding='same')(input_img)
    x = ELU()(x)
    x = Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)

    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)

    x = Conv2D(32, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = Conv2D(32, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)

    x = Conv2D(16, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = Conv2D(16, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)
    # 2x2x8
    encoder = Flatten(name='encoding_layer')(x)
    return Model(input_img, encoder)

def _get_decoder(input_shape=(64,)):
    input_code = Input(input_shape, name='decoder_input')
    x = Reshape(target_shape=(2, 2, 16))(input_code)

    x = Deconv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = ELU()(x)
    x = Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization()(x)

    x = Deconv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = ELU()(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization()(x)

    x = Deconv2D(32, kernel_size=3, strides=2, padding='same')(x)
    x = ELU()(x)
    x = Conv2D(32, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)

    x = Deconv2D(16, kernel_size=3, strides=2, padding='same')(x)
    x = ELU()(x)
    x = Conv2D(16, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization()(x)

    decoder = Conv2D(1, kernel_size=5, strides=1, padding='valid', activation='sigmoid', name='decoding_layer')(x)
    return Model(input_code, decoder)

def get_autoencoder_model(input_shape=(28, 28, 1), print_summary=False):
    x = Input(shape=input_shape)
    encoder = _get_encoder()
    decoder = _get_decoder()
    model = Model(x, decoder(encoder(x)))
    if print_summary:
        encoder.summary()
        decoder.summary()
        model.summary()
    return model, encoder, decoder

def freeze_model(model):
    for layer in model.layers:
        layer.trainable = False


def unfreeze_model(model):
    for layer in model.layers:
        layer.trainable = True


def evaluate_model(model_path, dataset_path = 'emnist/emnist-balanced-test.csv'):
    raw_test_x, raw_test_y, class_map = data_utils.load_dataset(dataset_path)
    test_x, test_y, _ = data_utils.prepare_data(raw_test_x, raw_test_y, class_map)
    best_model = load_model(model_path)
    print(best_model.evaluate(test_x, test_y))
    data_utils.print_confusion_matrix(test_x, test_y, model_path, class_map)

def prepare_classification_model(model):
    try:
        featurizer = Model(model.input, model.get_layer(name='fc3').output)
    except Exception as e:
        featurizer = model
    freeze_model(featurizer)
    return featurizer

class AutoencoderCheckpointer(Callback):
    def __init__(self, directory, base_name, encoder, decoder, save_model=False):
        super().__init__()
        self.path_model = os.path.join(directory, base_name)
        self.path_encoder = os.path.join(directory, base_name + '_encoder')
        self.path_decoder = os.path.join(directory, base_name + '_decoder')

        self.encoder = encoder
        self.decoder = decoder
        self.best_loss = np.inf
        self.save_model = save_model

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_loss')
        if current < self.best_loss:
            self.best_loss = current
            if self.save_model:
                self.model.save(self.path_model + '_' + str(epoch) + '_' + "{:.2f}".format(current) + '.hdf5')
            self.encoder.save(self.path_encoder + '_' + str(epoch) + '_' + "{:.2f}".format(current) + '.hdf5')
            self.decoder.save(self.path_decoder + '_' + str(epoch) + '_' + "{:.2f}".format(current) + '.hdf5')

