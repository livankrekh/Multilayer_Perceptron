
import pandas as pd
import numpy as np

class Multilayer_Perceptron(object):
	def __init__(self, X, y, n_perceptrons=4, n_layers=2):
		self.X = X
		self.y = y
		self.n_p = n_perceptrons
		self.n_l = n_layers
		self.res = np.array([])
		self.bias = np.random.random(n_layers + 1)
		self.max = self.X.max()

		self.NN = [[0] * n_perceptrons] * n_layers
		self.input_layer = [0] * n_perceptrons
		self.output_layer = [0] * len(np.unique(self.y))

		for i in range(len(self.input_layer)):
			self.input_layer[i] = np.random.random(len(X.columns))

		for i in range(len(self.output_layer)):
			self.output_layer[i] = np.random.random(n_perceptrons)

		for i in range(len(self.NN)):
			for j in range(len(self.NN[i])):
				self.NN[i][j] = np.random.random(n_perceptrons)

	def scaling(self):
		self.X = self.X / self.X.max()

	def leaky_ReLu(self, X):
		return np.where(X >= 0, X, X * 0.01)

	def sigmoid(self, X):
		return 1 / (1 + np.exp(X * -1))

	def softmax(self, X):
		return np.exp(X) / sum(np.exp(X))

	def prop(self):
		self.res = [[0] * self.n_p] * (self.n_l + 1)

		for i in range(len(self.res[0])):
			self.res[0][i] = self.sigmoid(self.X.dot(self.input_layer[i]) + self.bias[0])

		self.res[0] = pd.DataFrame(self.res[0]).T

		for i in range(len(self.NN)):
			for j in range(len(self.NN[i])):
				self.res[i + 1][j] = self.sigmoid(self.res[i].dot(self.NN[i][j]) + self.bias[i + 1])
			self.res[i + 1] = pd.DataFrame(self.res[i + 1]).T

		last_layer = [0] * len(np.unique(self.y))

		for i in range(len(last_layer)):
			last_layer[i] = self.softmax(self.res[len(self.res) - 1].dot(self.output_layer[i]) + self.bias[len(self.bias) - 1])
			last_layer[i] = pd.DataFrame(last_layer[i])


	def backprop(self):
		pass
