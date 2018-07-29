from models import *
from data_utils import *
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
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


def cvae_train(data_path,
               featurizer_path,
               model_name,
               reconstruction,
               batch_size,
               epochs,
               limit_gpu_fraction,
               latent_dim=4):
    # Notify user what are we training
    model_base_path = os.path.join(os.path.join('models', 'cvae'), model_name)
    config = {'data_path': data_path, 'featurizer_path': featurizer_path, 'model_name': model_base_path,
              'reconstruction': reconstruction, 'batch_size': batch_size, 'epochs': epochs,
              'limit_gpu_fraction': limit_gpu_fraction, 'latent_dim': latent_dim}


    # Limit GPU memory
    _limit_gpu_memory(limit_gpu_fraction)

    # Load data
    x_train, y_train, class_map = load_dataset(data_path)
    x_train, y_train, n_class = prepare_data(x_train, y_train, class_map)
    condition_train = cvae_get_encodings(x_train, y_train, n_class, featurizer_path=featurizer_path)

    # Split data
    validation_split = 0.2
    validation_start_idx = int((1 - validation_split) * x_train.shape[0])
    np.random.seed(42)
    random_idxs = np.random.permutation(x_train.shape[0])

    x_validate = x_train[random_idxs[validation_start_idx:]]
    condition_validate = condition_train[random_idxs[validation_start_idx:]]
    y_validate = y_train[random_idxs[validation_start_idx:]]
    x_train = x_train[random_idxs[:validation_start_idx]]
    condition_train = condition_train[random_idxs[:validation_start_idx]]
    y_train = y_train[random_idxs[validation_start_idx:]]

    # Get cvae
    cvae, encoder, decoder = get_cvae((784, ), (28, 28, 1),
                                      condition_train.shape[1],
                                      latent_dim=latent_dim,
                                      reconstruction=reconstruction)

    # Run training
    cvae.fit([x_train, condition_train],
             epochs=epochs,
             batch_size=batch_size,
             validation_data=([x_validate, condition_validate], None),
             callbacks=[AutoencoderCheckpointer(model_base_path, model_name,
                                                encoder, decoder, config),
                        TensorBoard(os.path.join('logs', model_name))])

    # Plot training
    cvae_plot_results((encoder, decoder), (x_validate, y_validate, condition_validate), class_map, model_base_path,
                      model_name, latent_dim)



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


def visualize_CVAE(feature_gen_path, dec_path, data_path='emnist/emnist-balanced-train.csv'):
    # Load raw data, normalize and hot encode
    raw_train_x, raw_train_y, class_map = load_dataset(data_path)
    train_x_all, _, n_class = prepare_data(raw_train_x, raw_train_y, class_map)

    # Split data set to train/validation
    # NOTE: X is input and output this is not mistake
    train_x, validation_x, train_y, validation_y = train_test_split(train_x_all, train_x_all,
                                                                    test_size=0.2, random_state=42)
    # Evaluate CVAE
    decoder = load_model(dec_path)
    feature_vec_gen = load_model(feature_gen_path)
    feature_vec_gen = prepare_classification_model(feature_vec_gen)
    explore_x = train_x[:5]

    # Visualize auto encoder
    show_orig = []
    show_output = []
    latent_dim = 64
    for img in explore_x:
        c = feature_vec_gen .predict(img.reshape(1, 28, 28, 1))
        c = c.reshape((-1,))
        z = np.zeros((64, ))
        decoder_input = np.reshape(np.concatenate([z, c]), (1, 96))
        out_img = decoder.predict(decoder_input)

        show_orig.append(img.reshape((28, 28)))
        show_output.append(out_img.reshape((28, 28)))

    for i in range(len(show_orig)):
        show_orig[i] = (show_orig[i] + 1) / 2
        show_output[i] = (show_output[i] + 1) / 2

    show_images(show_orig + show_output, cols=2)
