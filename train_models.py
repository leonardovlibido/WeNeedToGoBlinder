from models import *
from data_utils import *
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
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
    model = get_autoencoder_model()
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
                                  callbacks=[ModelCheckpoint(save_best_only=True,
                                                             filepath=checkpoint_path),
                                             ReduceLROnPlateau(factor=0.2, verbose=1),
                                             TensorBoard(log_dir='logs')])
    plot_history(history)
