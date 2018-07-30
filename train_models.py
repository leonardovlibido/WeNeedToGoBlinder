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
               encoding_type,
               batch_size,
               epochs,
               limit_gpu_fraction,
               latent_dim=4):
    # Notify user what are we training
    model_base_path = os.path.join(os.path.join('models', 'cvae'), model_name)
    config = {'data_path': data_path, 'featurizer_path': featurizer_path, 'model_name': model_base_path,
              'reconstruction': reconstruction, 'encoding_type':encoding_type, 'batch_size': batch_size, 'epochs': epochs,
              'limit_gpu_fraction': limit_gpu_fraction, 'latent_dim': latent_dim}


    # Limit GPU memory
    _limit_gpu_memory(limit_gpu_fraction)

    # Load data
    x_train, y_train, class_map = load_dataset(data_path)
    x_train, y_train, n_class = prepare_data(x_train, y_train, class_map)
    condition_train = cvae_get_encodings(x_train, y_train, n_class, encoding_type, featurizer_path=featurizer_path)

    x_test, y_test, _ = load_dataset('emnist/emnist-balanced-test.csv')
    x_test, y_test, _ = prepare_data(x_test, y_test, class_map)
    condition_test = cvae_get_encodings(x_test, y_test, n_class, encoding_type, featurizer_path=featurizer_path)

    # Get cvae
    cvae, encoder, decoder = get_cvae((784, ), (28, 28, 1),
                                      condition_train.shape[1],
                                      latent_dim=latent_dim,
                                      reconstruction=reconstruction)

    # Run training
    cvae.fit([x_train, condition_train],
             epochs=epochs,
             batch_size=batch_size,
             validation_data=([x_test, condition_test], None),
             callbacks=[AutoencoderCheckpointer(model_base_path, model_name,
                                                encoder, decoder, config),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10),
                        TensorBoard(os.path.join('logs', model_name))])

    # Plot training
    cvae_plot_results((encoder, decoder), (x_test, y_test, condition_test), class_map, model_base_path,
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


def cvae_visualize(data_path, featurizer_path, cvae_decoder_path):
    # Load data
    x_train, y_train, class_map = load_dataset(data_path)
    x_train, y_train, n_class = prepare_data(x_train, y_train, class_map)

    # Load models
    featurizer = load_model(featurizer_path)
    decoder = load_model(cvae_decoder_path)

    # Chose random example from each class
    x_samples = []
    y_samples = []
    np.random.seed(42)
    for class_idx in range(n_class):
        indexes = np.argmax(y_train, axis=1) == class_idx
        x_class = x_train[indexes]
        y_class = y_train[indexes]

        sample_idx = np.random.randint(low=0, high=x_class.shape[0])
        x_samples.append(x_class[sample_idx])
        y_samples.append(y_class[sample_idx])

    # Form dataset
    x_samples = np.array(x_samples)
    y_samples = np.array(y_samples)
    condition_samples = featurizer.predict(np.reshape(x_samples, (-1, 28, 28, 1)))

    # Call visualization
    decoder_fname = os.path.basename(cvae_decoder_path)
    decoder_parts = decoder_fname.split('_')
    model_name = ''
    for part in decoder_parts:
        if part == 'decoder':
            break

        if model_name == '':
            model_name = part
        else:
            model_name = model_name + '_' + part

    # Check model base path
    model_base_path = os.path.join('visualizations', model_name)
    if not os.path.exists(model_base_path):
        print('Creating directory for visualizations: ' + model_base_path)
        os.makedirs(model_base_path)

    # Plot originals
    for idx in range(n_class):
        # Get class label and sample
        sample = x_samples[idx]
        condition = condition_samples[idx]

        # Plot original
        fig = plt.figure()
        plt.imshow(np.reshape(sample, (28, 28)), cmap='Greys_r')
        plt.savefig(os.path.join(model_base_path,
                             model_name + '_' + str(idx) + '_' + 'orig' + '.png'))
        plt.close(fig)

        # Get generated
        latent_dim = 4
        decoder_in = np.hstack((np.zeros((latent_dim, )), condition))
        sample_generated = decoder.predict(np.array([decoder_in]))

        # Plot generated
        fig = plt.figure()
        plt.imshow(np.reshape(sample_generated, (28, 28)), cmap='Greys_r')
        plt.savefig(os.path.join(model_base_path,
                                 model_name + '_' + str(idx) + '_' + 'gen' + '.png'))
        plt.close(fig)
