import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import enum

# np.random.seed(67)
# np.random.seed(8954)

class Layer(enum.IntEnum):
	sigmoid = 0
	tanh = 1
	relu = 2
	l_relu = 3
	softmax = 4

class Multilayer_Perceptron(object):
	def __init__(self, X, y, layers=[Layer.sigmoid, Layer.sigmoid, Layer.sigmoid] ,n_perceptrons=[20,20], batch_size=20, alpha=0.04, validator=False, L1=0.0, L2=0.0):
		self.X = X
		self.y = y
		self.y_all = y
		self.y_name = np.unique(self.y)
		self.n_p = n_perceptrons
		self.res = []
		self.bias = []
		self.alpha = alpha
		self.NN = []
		self.output_layer = []
		self.b_s = batch_size
		self.valid = validator
		self.layers = layers
		self.losses = []
		self.val_losses = []
		self.results = []

		self.X_test = []
		self.y_test = []

		self.activation = [self.sigmoid, self.tanh, self.reLu, self.leaky_ReLu, self.softmax]
		self.derivative = [self.sigmoid_derivative, self.tanh_derivative, self.reLu_derivative, self.leaky_ReLu_derivative, self.sigmoid_derivative]

		self.l1 = L1
		self.l2 = L2

		for i in range(len(self.y_name)):
			self.output_layer.append(np.random.uniform(-0.5, 0.5, size=n_perceptrons[-1]))

		for i in range(len(n_perceptrons)):
			tmp = []
			self.bias.append(pd.Series(np.random.uniform(-0.5, 0.5, size=n_perceptrons[i])))
			for j in range(n_perceptrons[i]):
				if (i == 0):
					tmp.append(np.random.uniform(-0.5, 0.5, size=len(self.X.columns)))
				else:
					tmp.append(np.random.uniform(-0.5, 0.5, size=n_perceptrons[i - 1]))

			self.NN.append(tmp)

		self.NN.append(self.output_layer)
		self.bias.append(pd.Series(np.random.uniform(-0.5,0.5, size=len(self.y_name))))

	def data_split(self, train_size=0.8):
		X_new = self.X.sample(frac=train_size, replace=True)
		y_new = self.y.iloc[X_new.index]

		X_test = self.X.drop(X_new.index)
		y_test = self.y.drop(y_new.index)

		self.X = pd.DataFrame(X_new.values)
		self.y_all = pd.Series(y_new.values)
		self.y = self.y_all

		self.X_test = pd.DataFrame(X_test.values)
		self.y_test = pd.Series(y_test.values)

	def scaling(self):
		for i in self.X.columns:
			self.X[i] = (self.X[i] - self.X[i].mean()) / self.X[i].std()

	def reLu(self, X):
		return np.where(X >= 0, X, 0)

	def leaky_ReLu(self, X):
		return np.where(X >= 0, X, X * 0.01)

	def sigmoid(self, X):
		return 1 / (1 + np.exp(X * -1))

	def softmax(self, x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum(axis=0)

	def tanh(self, X):
		return (np.exp(X) - np.exp(X * -1)) / (np.exp(X) + np.exp(X * -1))

	def tanh_derivative(self, X):
		return 1 - (self.tanh(X) ** 2)

	def sigmoid_derivative(self, X):
		return self.sigmoid(X) * (1 - self.sigmoid(X))

	def reLu_derivative(self, X):
		return np.where(X >= 0, 1, 0)

	def leaky_ReLu_derivative(self, X):
		return np.where(X >= 0, 1, 0.01)

	def viz_loss(self):
		plt.plot(self.results, label="results")
		plt.plot(self.val_losses, label="value loss")
		plt.ylabel("Loss")
		plt.xlabel("Epoch")
		plt.legend()
		plt.show()

		plt.plot(self.losses, label="loss")
		plt.ylabel("Loss")
		plt.xlabel("Epoch")
		plt.legend()
		plt.show()

	def val_loss(self):
		if (len(self.res) < 1):
			return -1

		res = 0.0
		l = 0

		for i in range(len(self.y_name)):
			y = np.where(self.y == self.y_name[i], 1, 0)
			l += len(y)
			for j in range(len(y)):
				res += (y[j] * np.log(self.res[-1][i][j])) + ((1 - y[j]) * (1 - self.res[-1][i][j]))

		res = (res / l) * -1

		self.val_losses.append(res)

		return res

	def loss(self):
		self.results.append(self.test() / 100)

		if (len(self.res) < 1):
			return -1

		res = 0.0
		l = 0

		for i in range(len(self.y_name)):
			y = np.where(self.y_test == self.y_name[i], 1, 0)
			l += len(y)
			for j in range(len(y)):
				res += (y[j] * np.log(self.res[-1][i][j])) + ((1 - y[j]) * (1 - self.res[-1][i][j]))

		res = (res / l) * -1

		self.losses.append(res)

		return res

	def test(self):
		output = np.array(self.predict(self.X_test))
		y = np.array(self.y_test)

		for i in range(len(y)):
			output[i] = self.y_name.tolist().index(output[i])
			y[i] = self.y_name.tolist().index(y[i])

		res = output.astype(int) - y.astype(int)
		res = np.where(res == 0, 1, 0)

		return int(sum(res) / len(res) * 100)

	def train(self, epoch=100, print_loss=True):
		for i in range(epoch):
			if print_loss:
				print("Epoch ", i + 1, "/", epoch, ", val_loss - %.5f"% (self.val_loss()),  ", loss - %.5f"% (self.loss()), sep='')
			self.prop()
			self.backprop()

	def predict(self, X=[]):
		self.res = []

		if (len(X) < 1):
			X = self.X

		self.res.append(X)

		self.res[-1] = pd.DataFrame(self.res[-1].values)

		for i in range(len(self.NN) - 1):
			tmp = []
			for j in range(len(self.NN[i])):
				f_in = self.res[i].dot(self.NN[i][j]) + self.bias[i][j]
				tmp.append(self.activation[self.layers[i]](f_in))
			self.res.append(pd.DataFrame(tmp).T)

		last_layer = []

		for i in range(len(self.y_name)):
			f_in = self.res[-1].dot(self.NN[-1][i]) + self.bias[-1][i]
			tmp = self.activation[self.layers[-1]](f_in)
			last_layer.append(pd.Series(tmp))

		self.res.append(pd.DataFrame(last_layer).T)
		res = []

		for i in range(len(self.res[-1].iloc[:,0])):
			if (sum(self.res[-1].iloc[i]) >= 1.5 and self.valid):
				res.append(None)
			else:
				label_j = 0

				for j in range(len(self.y_name)):
					if (self.res[-1].iloc[i, j] > self.res[-1].iloc[i, label_j]):
						label_j = j

				res.append(self.y_name[label_j])

		return (res)

	def prop(self):
		self.res = []
		self.res.append(self.X.sample(self.b_s))
		self.y = self.y_all.iloc[self.res[-1].index]

		self.res[-1] = pd.DataFrame(self.res[-1].values)

		for i in range(len(self.NN) - 1):
			tmp = []
			for j in range(len(self.NN[i])):
				f_in = self.res[i].dot(self.NN[i][j]) + self.bias[i][j]
				tmp.append( self.activation[self.layers[i]](f_in) )
			self.res.append(pd.DataFrame(tmp).T)

		last_layer = []

		for i in range(len(self.y_name)):
			f_in = self.res[-1].dot(self.NN[-1][i]) + self.bias[-1][i]
			tmp = self.activation[self.layers[-1]](f_in)
			last_layer.append(pd.Series(tmp))

		self.res.append(pd.DataFrame(last_layer).T)

	def backprop(self):
		prev_error = []

		for i in range(len(self.y_name)):
			y = np.where(self.y == self.y_name[i], 1, 0)

			prev_error.append(self.res[-1][i] - pd.Series(y))

		for i, nn in enumerate(reversed(self.NN)):
			tmp = []

			for _ in range(len(nn[0])):
				tmp.append(0.0)

			for j in range(len(nn)):
				fd_in = self.res[(i + 2) * -1].dot(nn[j]) + self.bias[(i + 1) * -1][j]
				o_k = prev_error[j] * self.derivative[self.layers[(i + 1) * -1]](fd_in)
				self.bias[(i + 1) * -1][j] -= sum(self.alpha * o_k)

				for k in range(len(nn[j])):
					current = self.res[(i + 2) * -1]
					w = self.alpha * o_k * current[current.columns[k]]
					nn[j][k] -= sum(w) + (self.l1 * abs(nn[j][k])) + (self.l2 * nn[j][k] ** 2)
					tmp[k] += sum(o_k * nn[j][k])

			prev_error = tmp

	def save(self, path):
		general = {"nn": self.NN, "y_name": self.y_name}
		np.save(path, general)
		print("\033[1m\033[32mModel saved to '" + path + ".npy'\033[0m")

