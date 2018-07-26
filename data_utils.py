import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import csv
from skimage.filters import gaussian
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

def load_dataset(fpath='emnist/emnist-balanced-train.csv', mpath='emnist/emnist-balanced-mapping.txt'):
	X = []
	Y = []

	M = {}
	with open(fpath) as csv_file:
		reader = csv.reader(csv_file)
		for idx, row in enumerate(reader):
			Y.append(int(row[0]))
			X.append(np.transpose(np.reshape(np.array(row[1:], np.float32), (28, 28, 1)), [1, 0, 2]))
	X = np.array(X)
	Y = np.array(Y)

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


def explore_dataset(X, Y, M, n_data, n_samples=5):
	for idx in np.random.choice(n_data, n_samples):
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
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()

	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()


def print_confusion_matrix(dataset_x, dataset_y, model_path):
	_, _, class_map = load_dataset()
	labels = []
	for i in range(0,47):
		labels.append(class_map[i])

	best_model = load_model(model_path)
	predictions = best_model.predict(dataset_x)

	predictions_nums = np.argmax(predictions, axis=1)
	dataset_y = np.argmax(dataset_y, axis=1)

	confusion_mat = confusion_matrix(dataset_y, predictions_nums)

	print('  ', end='  ')
	for j in labels:
		print(str(j), end='   ')
	print()
	for idx, i in enumerate(confusion_mat):
		print(str(labels[idx]), end=' ')
		for j in i:
			print('%3d' % (j), end=' ')
		# print(j, end=' ')
		print()


class ElasticDistortion:
	def __init__(self, grid_shape, alpha=50, sigma=1.5):
		self.x_grid_dim, self.y_grid_dim = grid_shape
		self.x, self.y = np.meshgrid(np.arange(self.y_grid_dim), np.arange(self.x_grid_dim))
		self.alpha = alpha
		self.sigma = sigma

	def _bilinear_interpolation(self, img, x, y):
		x1 = np.floor(x).astype(np.int32)
		x2 = x1 + 1
		y1 = np.floor(y).astype(np.int32)
		y2 = y1 + 1
		x1 = np.clip(x1, a_min=0, a_max=img.shape[1] - 1)
		x2 = np.clip(x2, a_min=0, a_max=img.shape[1] - 1)
		y1 = np.clip(y1, a_min=0, a_max=img.shape[0] - 1)
		y2 = np.clip(y2, a_min=0, a_max=img.shape[0] - 1)
		r1 = (x2 - x) * img[y1, x1] + (x - x1) * img[y1, x2]
		r2 = (x2 - x) * img[y2, x1] + (x - x1) * img[y2, x2]
		return (y - y1) * r2 + (y2 - y) * r1

	def __call__(self, in_img):
		img = np.reshape(in_img, (in_img.shape[0], in_img.shape[1]))

		dx = 2 * np.random.rand(*img.shape) - 1
		dy = 2 * np.random.rand(*img.shape) - 1
		dx = dx / np.linalg.norm(dx)
		dy = dy / np.linalg.norm(dy)

		fdx = gaussian(dx, sigma=self.sigma)
		fdy = gaussian(dy, sigma=self.sigma)

		sdx = self.alpha * fdx
		sdy = self.alpha * fdy

		indices = np.reshape(self.y + sdy, (-1, 1)), np.reshape(self.x + sdx, (-1, 1))
		img = self._bilinear_interpolation(img, indices[1], indices[0])
		return np.reshape(img, in_img.shape)