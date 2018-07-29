from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Dense, Flatten, Input, ELU
from keras.layers import Deconv2D, Reshape, Concatenate, Lambda
from keras.models import Model
import data_utils
from keras.models import load_model
from keras.callbacks import Callback
import os
import numpy as np
from keras import backend as K
from keras.losses import binary_crossentropy, mse
import matplotlib.pyplot as plt
import json


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

def _cvae_sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def _cvae_loss(X, outputs, z_mean, z_log_var, reconstruction):
    if reconstruction == 'binary_crossentropy':
        reconstruction_loss = binary_crossentropy(X, outputs)
    elif reconstruction == 'mse':
        reconstruction_loss = mse(X, outputs)
    else:
        raise ValueError('Reconstruction losses are crossentropy or mse.')

    reconstruction_loss *= K.cast(K.shape(X)[1], 'float32')
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    cvae_loss = K.mean(reconstruction_loss + kl_loss)
    return cvae_loss, reconstruction_loss, kl_loss


def cvae_get_encodings(train_x_norm,
                       train_y_hot,
                       n_class,
                       featurizer_path=None,
                       featurizer=None):
    # Load feature vector generator and decoder
    if featurizer_path is None and featurizer is None:
        raise ValueError('You must set either featurizer of featurizer_path')
    if featurizer_path is not None:
        assert featurizer == None, 'Featurizer!'
        featurizer = load_model(featurizer_path)

    # Get all encodings
    predictions = featurizer.predict(train_x_norm.reshape(-1, 28, 28, 1))

    # Init variables used to obtain mean vector for every class
    encoding_dim = predictions.shape[1]
    feature_vec_means = np.zeros((n_class, encoding_dim))
    train_y = np.argmax(train_y_hot, axis=1)
    encodings = np.zeros((train_x_norm.shape[0], encoding_dim))

    # Mean encodings
    for i in range(n_class):
        indexes = train_y == i
        feature_vec = np.mean(predictions[indexes], axis=0)
        feature_vec_means[i] = feature_vec
        encodings[indexes] = feature_vec

    return encodings


def get_cvae(input_shape, img_shape, condition_dim,
             latent_dim, print_summary=False,
             reconstruction='binary_crossentropy'):
    # Inputs
    X = Input(shape=input_shape, name='encoder_input')
    label = Input(shape=(condition_dim,))
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

    # Sampling
    z = Lambda(_cvae_sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    zc = Concatenate()([z, label])

    # instantiate encoder model
    encoder = Model([X, label], [z_mean, z_log_var, zc], name='encoder')
    encoder.summary()

    # Decoder
    latent_inputs = Input(shape=(latent_dim + condition_dim,), name='z_sampling')
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
    cvae = Model([X, label], outputs, name='cvae_64')

    # Add model loss
    loss, reconstruction_loss, kl_loss = _cvae_loss(X, outputs, z_mean, z_log_var, reconstruction)
    cvae.add_loss(loss)
    cvae.compile(optimizer='adam', metrics=[reconstruction_loss, kl_loss])

    # Print summary
    if print_summary:
        encoder.summary()
        decoder.summary()
        cvae.summary()
    return cvae, encoder, decoder

def cvae_plot_results(models,
                      data,
                      class_map,
                      model_name='cvae'):

    encoder, decoder = models
    x_validation, y_validation, condition_validation = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, model_name+'.png')
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

    # filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    idx = np.random.randint(low=0, high=x_validation.shape[0])
    label = condition_validation[idx]
    print(class_map[np.argmax(y_validation[idx])])

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


def _get_encoder(input_shape=(28, 28, 1)):
    input_img = Input(shape=input_shape, name='encoder_input')

    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(input_img)
    x = ELU()(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)

    x = Conv2D(32, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = Conv2D(32, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)

    x = Conv2D(16, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = Conv2D(16, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)

    x = Conv2D(8, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = Conv2D(8, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)
    # 2x2x8
    encoder = Flatten(name='encoding_layer')(x)
    return Model(input_img, encoder)

def _get_decoder(input_shape=(32,)):
    input_code = Input(input_shape, name='decoder_input')
    x = Reshape(target_shape=(2, 2, 8))(input_code)

    x = Deconv2D(8, kernel_size=3, strides=2, padding='same')(x)
    x = ELU()(x)
    x = Conv2D(8, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)

    x = Deconv2D(16, kernel_size=3, strides=2, padding='same')(x)
    x = ELU()(x)
    x = Conv2D(16, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)

    x = Deconv2D(32, kernel_size=3, strides=2, padding='same')(x)
    x = ELU()(x)
    x = Conv2D(32, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)

    x = Deconv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = ELU()(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = ELU()(x)

    decoder = Conv2D(1, kernel_size=5, strides=1, padding='valid', activation='tanh', name='decoding_layer')(x)
    return Model(input_code, decoder)

def get_autoencoder_model(input_shape=(28, 28, 1), print_summary=False):
    x = Input(shape=input_shape)
    encoder = _get_encoder()
    decoder = _get_decoder()
    model = Model(x, decoder(encoder(x)))
    if print_summary:
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
    # featurizer = Model(model.input, model.get_layer(name='fc3').output)
    featurizer = model
    freeze_model(featurizer)
    return featurizer

class AutoencoderCheckpointer(Callback):
    def __init__(self, directory, base_name, encoder, decoder, config, save_model=False):
        super().__init__()
        if os.path.isdir(directory):
            print("Checkpoint: found directory, skipping creation")
        elif os.path.exists(directory):
            raise ValueError('Checkpoint: path exists but it is not directory')
        else:
            os.makedirs(directory)
            print('Checkpoint: created directory')
        with open(os.path.join(directory, base_name + '.txt'), 'w') as config_file:
            config_file.write(json.dumps(config, indent=4))

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

