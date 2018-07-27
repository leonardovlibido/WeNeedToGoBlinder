import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from data_utils import *
from models import *


N_BALANCED_TRAIN = 112800
INPUT_SHAPE = (28, 28, 1)
CLASS_BATCH_SIZE = 64


# raw_train_x, raw_train_y, class_map = load_dataset()
# # explore_dataset(train_x, train_y, train_map)
# train_x_all, train_y_all, n_class = prepare_data(raw_train_x, raw_train_y, class_map)
# train_x, validation_x, train_y, validation_y = train_test_split(train_x_all, train_y_all, test_size=0.2, random_state=42)
# model = get_classification_model(n_class, print_summary=True)
#
# model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
# datagen = ImageDataGenerator(shear_range=0.05, rotation_range=5, width_shift_range=0.1,
# 							 preprocessing_function=ElasticDistortion(grid_shape=(28, 28)))
# history = model.fit_generator(datagen.flow(train_x, train_y, CLASS_BATCH_SIZE),
# 							  epochs=100,
# 							  steps_per_epoch=N_BALANCED_TRAIN // CLASS_BATCH_SIZE,
# 							  validation_data=(validation_x, validation_y),
# 							  callbacks=[ ModelCheckpoint(save_best_only=True,
# 														  filepath='checkpoints/aug_model.{epoch:02d}-{val_loss:.2f}.hdf5'),
# 								ReduceLROnPlateau(factor=0.2, verbose=1)])
# plot_history(history)
#
# prepare_classification_model(model)
#
# model.summary()

evaluate_model('best_model_classifier/aug_model.43-0.27.hdf5')
