from models import *
from data_utils import *
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import mse, binary_crossentropy
import keras.backend as K
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

CLASS_BATCH_SIZE = 64

def _limit_gpu_memory(limit_fraction):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = limit_fraction
    set_session(tf.Session(config=config))

def classifier_train(data_path='emnist/emnist-balanced-train.csv',
                     batch_size=CLASS_BATCH_SIZE,
                     epochs=50,
                     model_checkpoint_dir='checkpoints',
                     model_checkpoint_name='final_32',
                     limit_gpu_fraction=0.3):
    # Limit GPU memory if not None
    if limit_gpu_fraction is not None:
        _limit_gpu_memory(limit_gpu_fraction)

    # Load raw data, normalize and hot encode
    raw_train_x, raw_train_y, class_map = load_dataset(data_path)
    train_x_all, train_y_all, n_class = prepare_data(raw_train_x, raw_train_y, class_map)

    # Split data set to train/validation
    train_x, validation_x, train_y, validation_y = train_test_split(train_x_all, train_y_all,
                                                                    test_size=0.2, random_state=42)

    # Get model for classification and compile it
    model = get_classification_model(n_class)
    model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

    # Create image generator
    datagen = ImageDataGenerator(shear_range=0.05, rotation_range=5, width_shift_range=0.1,
                                 preprocessing_function=ElasticDistortion(grid_shape=(28, 28)))

    # Fit model and plot history
    checkpoint_path = os.path.join(model_checkpoint_dir, model_checkpoint_name + '.{epoch:02d}-{val_loss:.2f}.hdf5')
    history = model.fit_generator(datagen.flow(train_x, train_y, batch_size),
                                  epochs=epochs,
                                  steps_per_epoch=train_x.shape[0] // batch_size,
                                  validation_data=(validation_x, validation_y),
                                  callbacks=[ModelCheckpoint(save_best_only=True,
                                                             filepath=checkpoint_path),
                                             ReduceLROnPlateau(factor=0.2, verbose=1),
                                             TensorBoard(log_dir='logs')])
    plot_history(history)


def CVAE_train(data_path='emnist/emnist-balanced-train.csv',
               featurizer_path='best_model_classifier/aug_32.43-0.28.hdf5',
               batch_size=CLASS_BATCH_SIZE,
               epochs=50,
               model_checkpoint_dir='CVAE_checkpoints',
               model_checkpoint_name='CVAE_32',
               limit_gpu_fraction=0.5):
    # Limit GPU memory if not None
    if limit_gpu_fraction is not None:
        _limit_gpu_memory(limit_gpu_fraction)

    # Load raw data, normalize and hot encode
    raw_train_x, raw_train_y, class_map = load_dataset(data_path)
    train_x_all, _, n_class = prepare_data(raw_train_x, raw_train_y, class_map)

    # Split data set to train/validation
    # NOTE: X is input and output this is not mistake
    train_x, validation_x, train_y, validation_y = train_test_split(train_x_all, train_x_all,
                                                                    test_size=0.2, random_state=42)

    # Get model for classification and compile it
    feature_vec_gen = load_model(featurizer_path)
    feature_vec_gen = prepare_classification_model(feature_vec_gen)

    model, encoder, decoder, z_mean, z_log_sigma = get_CVAE_model(feature_vec_gen, print_summary=True)

    def vae_loss(y_true, y_pred):
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)
        recon = K.mean(K.square(y_true_flat - y_pred_flat))
        kl = 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mean) - 1. - z_log_sigma)
        return recon + kl

    def KL_loss(y_true, y_pred):
        return 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mean) - 1. - z_log_sigma)
    #     return 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mean) - 1. - z_log_sigma, axis=1)
    #
    def recon_loss(y_true, y_pred):
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)
        return K.mean(K.square(y_true_flat - y_pred_flat))
    #     y_true_flat, y_pred_flat = _remap_y(y_true, y_pred)
    #     return K.sum(K.binary_crossentropy(y_true_flat, y_pred_flat), axis=-1)

    model.compile(optimizer='adam', loss=vae_loss, metrics=[KL_loss, recon_loss])
    # model.compile(optimizer='adam', loss=KL_loss)

    # Create image generator
    datagen = ImageDataGenerator(shear_range=0.05, rotation_range=5, width_shift_range=0.1,
                                 preprocessing_function=ElasticDistortion(grid_shape=(28, 28)))

    # Fit model and plot history
    history = model.fit_generator(datagen.flow(train_x, train_y, batch_size),
                                  epochs=epochs,
                                  steps_per_epoch=train_x.shape[0] // batch_size,
                                  validation_data=(validation_x, validation_y),
                                  callbacks=[AutoencoderCheckpointer(model_checkpoint_dir, model_checkpoint_name,
                                                                     encoder, decoder),
                                             ReduceLROnPlateau(factor=0.2, verbose=1),
                                             TensorBoard(log_dir='logs/CVAE')])
    plot_history(history, have_accuracy=False)

    # Evaluate autoencoder
    # explore_x = train_x[:5]

    # Visualize auto encoder
    # show_orig = []
    # show_output = []
    # for img in explore_x:
    #     show_orig.append(img.reshape((28, 28)))
    #     out_img = decoder.predict(encoder.predict(np.reshape(img, (1, 28, 28, 1))))
    #     show_output.append(out_img.reshape((28, 28)))
    #
    # for i in range(len(show_orig)):
    #     show_orig[i] = (show_orig[i] + 1) / 2
    #     show_output[i] = (show_output[i] + 1) / 2

    # show_images(show_orig + show_output, cols=2)


def autoencoder_train(data_path='emnist/emnist-balanced-train.csv',
                      batch_size=CLASS_BATCH_SIZE,
                      epochs=50,
                      model_checkpoint_dir='autoenc_checkpoints',
                      model_checkpoint_name='autoenc_32',
                      limit_gpu_fraction=0.3):
    # Limit GPU memory if not None
    if limit_gpu_fraction is not None:
        _limit_gpu_memory(limit_gpu_fraction)

    # Load raw data, normalize and hot encode
    raw_train_x, raw_train_y, class_map = load_dataset(data_path)
    train_x_all, _, n_class = prepare_data(raw_train_x, raw_train_y, class_map)

    # Split data set to train/validation
    # NOTE: X is input and output this is not mistake
    train_x, validation_x, train_y, validation_y = train_test_split(train_x_all, train_x_all,
                                                                    test_size=0.2, random_state=42)

    # Get model for classification and compile it
    model, encoder, decoder = get_autoencoder_model()
    model.compile(optimizer='adam', loss='mse')

    # Create image generator
    datagen = ImageDataGenerator(shear_range=0.05, rotation_range=5, width_shift_range=0.1,
                                 preprocessing_function=ElasticDistortion(grid_shape=(28, 28)))

    # Fit model and plot history
    checkpoint_path = os.path.join(model_checkpoint_dir, model_checkpoint_name + '.{epoch:02d}-{val_loss:.2f}.hdf5')
    history = model.fit_generator(datagen.flow(train_x, train_y, batch_size),
                                  epochs=epochs,
                                  steps_per_epoch=train_x.shape[0] // batch_size,
                                  validation_data=(validation_x, validation_y),
                                  callbacks=[AutoencoderCheckpointer(model_checkpoint_dir, model_checkpoint_name,
                                                                     encoder, decoder),
                                      # ModelCheckpoint(save_best_only=True,
                                      #                        filepath=checkpoint_path),
                                             ReduceLROnPlateau(factor=0.2, verbose=1),
                                             TensorBoard(log_dir='logs/autoencoder')])
    plot_history(history, have_accuracy=False)

    # Evaluate autoencoder
    explore_x = train_x[:5]

    # Visualize auto encoder
    show_orig = []
    show_output = []
    for img in explore_x:
        show_orig.append(img.reshape((28, 28)))
        out_img = decoder.predict(encoder.predict(np.reshape(img, (1, 28, 28, 1))))
        show_output.append(out_img.reshape((28, 28)))

    for i in range(len(show_orig)):
        show_orig[i] = (show_orig[i] + 1) / 2
        show_output[i] = (show_output[i] + 1) / 2

    show_images(show_orig + show_output, cols=2)

def visualize_autoencoder(enc_path, dec_path, data_path='emnist/emnist-balanced-train.csv'):
    # Load raw data, normalize and hot encode
    raw_train_x, raw_train_y, class_map = load_dataset(data_path)
    train_x_all, _, n_class = prepare_data(raw_train_x, raw_train_y, class_map)

    # Split data set to train/validation
    # NOTE: X is input and output this is not mistake
    train_x, validation_x, train_y, validation_y = train_test_split(train_x_all, train_x_all,
                                                                    test_size=0.2, random_state=42)
    # Evaluate autoencoder
    encoder = load_model(enc_path)
    print("Encoder success!")
    decoder = load_model(dec_path)
    explore_x = train_x[:5]

    # Visualize auto encoder
    show_orig = []
    show_output = []
    for img in explore_x:
        show_orig.append(img.reshape((28, 28)))
        out_img = decoder.predict(encoder.predict(np.reshape(img, (1, 28, 28, 1))))
        show_output.append(out_img.reshape((28, 28)))

    for i in range(len(show_orig)):
        show_orig[i] = (show_orig[i] + 1) / 2
        show_output[i] = (show_output[i] + 1) / 2

    show_images(show_orig + show_output, cols=2)


