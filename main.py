import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from data_utils import *
from models import *


N_BALANCED_TRAIN = 112800
INPUT_SHAPE = (28, 28, 1)


raw_train_x, raw_train_y, class_map = load_dataset(N_BALANCED_TRAIN)
# explore_dataset(train_x, train_y, train_map)
train_x_all, train_y_all, n_class = prepare_data(raw_train_x, raw_train_y, class_map)
train_x, validation_x, train_y, validation_y = train_test_split(train_x_all, train_y_all, test_size=0.2, random_state=42)
model = get_classification_model(n_class, print_summary=True)

model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
history = model.fit(train_x, train_y, batch_size=32, epochs=75, validation_data=(validation_x, validation_y),
					callbacks=[ ModelCheckpoint(save_best_only=True,
												filepath='checkpoints/first_model.{epoch:02d}-{val_loss:.2f}.hdf5'),
								ReduceLROnPlateau(factor=0.2, verbose=1)])
plot_history(history)
