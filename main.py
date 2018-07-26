import matplotlib.pyplot as plt
import csv
import numpy as np
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Dense, Flatten, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


N_BALANCED_TRAIN = 112800
INPUT_SHAPE = (28, 28, 1)

def load_train(fpath='emnist/emnist-balanced-train.csv', mpath='emnist/emnist-balanced-mapping.txt'):
	X = np.zeros((N_BALANCED_TRAIN, 28, 28, 1), dtype=np.float32)
	Y = np.zeros((N_BALANCED_TRAIN, 1), dtype=np.int32)
	M = {}
	with open(fpath) as csv_file:
		reader = csv.reader(csv_file)
		for idx, row in enumerate(reader):
			Y[idx] = int(row[0])
			X[idx] = np.transpose(np.reshape(np.array(row[1:], np.float32), (28, 28, 1)), [1, 0, 2])

	with open(mpath) as map_file:
		for line in map_file:
			c_and_ascii = line.split()
			M[int(c_and_ascii[0].strip())] = str(chr(int(c_and_ascii[1].strip())))
	return X, Y, M


def prepare_data(X, Y, M, subtract_mean_img=False):
	n_class = len(list(M.keys()))
	X = 2 * (X / 255.) - 1
	Y = to_categorical(Y, n_class)
	return X, Y, n_class


def get_model(n_class, print_summary=False):
	x = Input(shape=INPUT_SHAPE)

	y_hat = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', name='conv1_1')(x)
	y_hat = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', name='conv1_2')(y_hat)
	y_hat = BatchNormalization(name='bn1')(y_hat)
	y_hat = MaxPool2D(pool_size=2, name='max_pool1')(y_hat)
	y_hat = Dropout(0.2, name='dropout1')(y_hat)

	y_hat = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv2_1')(y_hat)
	y_hat = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv2_2')(y_hat)
	y_hat = BatchNormalization(name='bn2')(y_hat)
	y_hat = MaxPool2D(pool_size=2, name='max_pool2')(y_hat)
	y_hat = Dropout(0.2, name='dropout2')(y_hat)

	y_hat = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name='conv3_1')(y_hat)
	y_hat = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name='conv3_2')(y_hat)
	y_hat = BatchNormalization(name='bn3')(y_hat)
	y_hat = MaxPool2D(pool_size=2, name='max_pool3')(y_hat)
	y_hat = Dropout(0.2, name='dropout3')(y_hat)

	y_hat = Flatten(name='flatten')(y_hat)
	y_hat = Dense(512, activation='relu', name='fc1')(y_hat)
	y_hat = Dropout(0.5, name='dropout4')(y_hat)
	y_hat = Dense(256, activation='relu', name='fc2')(y_hat)
	y_hat = Dropout(0.5, name='dropout5')(y_hat)
	y_hat = Dense(n_class, activation='softmax', name='softmax')(y_hat)

	model = Model(x, y_hat)
	if print_summary:
		model.summary()
	return model

def explore_dataset(X, Y, M, n_samples=5):
	for idx in np.random.choice(N_BALANCED_TRAIN, n_samples):
		print(M[Y[idx, 0]])
		plt.imshow(X[idx, :, :, 0], cmap='gray')
		plt.show()

def plot_history(history):
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

raw_train_x, raw_train_y, class_map = load_train()
# explore_dataset(train_x, train_y, train_map)
train_x_all, train_y_all, n_class = prepare_data(raw_train_x, raw_train_y, class_map)
train_x, validation_x, train_y, validation_y = train_test_split(train_x_all, train_y_all, test_size=0.2, random_state=42)
model = get_model(n_class, print_summary=True)

model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
history = model.fit(train_x, train_y, batch_size=32, epochs=75, validation_data=(validation_x, validation_y),
					callbacks=[ ModelCheckpoint(save_best_only=True,
												filepath='checkpoints/first_model.{epoch:02d}-{val_loss:.2f}.hdf5'),
								ReduceLROnPlateau(factor=0.2, verbose=1)])
plot_history(history)
