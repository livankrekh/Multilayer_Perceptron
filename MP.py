
import pandas as pd
import numpy as np

class Multilayer_Perceptron(object):
	def __init__(self, X, y, n_perceptrons=4, n_layers=2):
		self.X = X
		self.y = y
		self.y_name = np.unique(self.y)
		self.n_p = n_perceptrons
		self.n_l = n_layers
		self.res = []
		self.bias = np.random.random(n_layers + 1)
		self.alpha = 0.01

		self.NN = []
		self.output_layer = [0] * len(np.unique(self.y))

		for i in range(len(self.output_layer)):
			self.output_layer[i] = np.random.random(n_perceptrons)

		for i in range(n_layers):
			tmp = []
			for j in range(n_perceptrons):
				if (i == 0):
					tmp.append(np.random.random(len(self.X.columns)))
				else:
					tmp.append(np.random.random(n_perceptrons))

			self.NN.append(tmp)

	def scaling(self):
		for i in self.X.columns:
			self.X[i] = (self.X[i] - self.X[i].mean()) / self.X[i].std()

	def leaky_ReLu(self, X):
		return np.where(X >= 0, X, X * 0.01)

	def sigmoid(self, X):
		return 1 / (1 + np.exp(X * -1))

	def sigmoid_derivative(self, X):
		return self.sigmoid(X) * (1 - self.sigmoid(X))

	def softmax(self, X):
		return np.exp(X) / sum(np.exp(X))

	def softmax_derivative(self, X):
		s = np.array(X).reshape(-1,1)
		return np.diagflat(s) - np.dot(s, np.transpose(s))

	def prop(self):
		self.res.append(self.X)

		for i in range(len(self.NN)):
			tmp = []
			for j in range(len(self.NN[i])):
				tmp.append(self.sigmoid(self.res[i].dot(self.NN[i][j]) + self.bias[i]))
			self.res.append(pd.DataFrame(tmp).T)

		last_layer = []

		for i in range(len(self.y_name)):
			tmp = self.sigmoid(self.res[-1].dot(self.output_layer[i]) + self.bias[-1])
			last_layer.append(pd.DataFrame(tmp))

		self.res.append(last_layer)
		self.NN.append(self.output_layer)

	def backprop(self):
		prev_error = []

		for i in range(len(self.y_name)):
			y = np.where(self.y == self.y_name[i], 1, 0)

			prev_error.append(self.res[-1][i][0] - pd.Series(y))

		# print(prev_error)

		for i, nn in enumerate(reversed(self.NN)):
			tmp = []

			for _ in range(len(nn[0])):
				tmp.append(0.0)

			for j in range(len(nn)):
				o_k = prev_error[j] * self.sigmoid_derivative(self.res[(i + 2) * -1].dot(nn[j]) + self.bias[(i + 1) * -1])

				for k in range(len(nn[j])):
					w = self.alpha * o_k * self.res[(i + 2) * -1][k]
					nn[j][k] += sum(w)
					tmp[k] += sum(o_k * nn[j][k])

			prev_error = tmp

				# for k in range(len(prev_error)):

			# for j in range(len(prev_error)):
			# 	err = prev_error[j] * self.sigmoid_derivative(self.res[(i + 2) * -1].dot(nn[j]) + self.bias[(i + 1) * -1])
			# 	err = pd.DataFrame(err)
			# 	o_k = err.dot(pd.DataFrame(nn[j]).T)

			# 	print(o_k)

