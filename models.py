from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Dense, Flatten, Input
from keras.models import Model

def get_classification_model(n_class, input_shape=(28, 28, 1), print_summary=False):
	x = Input(shape=input_shape)

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