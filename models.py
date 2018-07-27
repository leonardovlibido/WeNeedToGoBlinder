from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Dense, Flatten, Input, ELU
from keras.models import Model
import data_utils
from keras.models import load_model

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
	model.layers.pop()
	model.layers.pop()
	freeze_model(model)
	model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
