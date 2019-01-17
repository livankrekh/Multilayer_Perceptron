import pandas as pd
import numpy as np

# np.random.seed(15)
# np.random.seed(8954) 41-47

class Multilayer_Perceptron(object):
	def __init__(self, X, y, n_perceptrons=[4], n_output=2):
		self.X = X
		self.y = y
		self.y_name = np.unique(self.y)
		self.n_p = n_perceptrons
		self.res = []
		self.bias = []
		self.alpha = 0.04
		self.NN = []
		self.output_layer = []

		for i in range(len(self.y_name)):
			self.output_layer.append(np.random.uniform(-0.5, 0.5, size=n_perceptrons[-1]))

		for i in range(len(n_perceptrons)):
			tmp = []
			self.bias.append(pd.Series(np.random.uniform(-0.5, 0.5, size=n_perceptrons[i])))
			for j in range(n_perceptrons[i]):
				if (i == 0):
					# tmp.append(np.array([1] * len(self.X.columns)))
					tmp.append(np.random.uniform(-0.5, 0.5, size=len(self.X.columns)))
				else:
					tmp.append(np.random.uniform(-0.5, 0.5, size=n_perceptrons[i - 1]))

			self.NN.append(tmp)

		self.NN.append(self.output_layer)
		self.bias.append(pd.Series(np.random.uniform(-0.5,0.5, size=len(self.y_name))))

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

	def loss(self):
		if (len(self.res) < 1):
			return -1

		res = 0.0
		l = 0

		for i in range(len(self.y_name)):
			y = np.where(self.y == self.y_name[i], 1, 0)
			l += len(y)
			for j in range(len(y)):
				res += (y[j] * np.log(self.res[-1][i][j])) + ((1 - y[j]) * (1 - self.res[-1][i][j]))

		return (res / l) * -1

	def prop(self):
		self.res = []
		self.res.append(self.X)

		for i in range(len(self.NN) - 1):
			tmp = []
			for j in range(len(self.NN[i])):
				tmp.append(self.sigmoid(self.res[i].dot(self.NN[i][j]) + self.bias[i][j]))
			self.res.append(pd.DataFrame(tmp).T)

		last_layer = []

		for i in range(len(self.y_name)):
			tmp = self.sigmoid(self.res[-1].dot(self.NN[-1][i]) + self.bias[-1][i])
			last_layer.append(pd.Series(tmp))

		self.res.append(pd.DataFrame(last_layer).T)

	def backprop(self):
		prev_error = []

		for i in range(len(self.y_name)):
			y = np.where(self.y == self.y_name[i], 1, 0)

			prev_error.append(self.res[-1][i][0] - pd.Series(y))

		for i, nn in enumerate(reversed(self.NN)):
			tmp = []

			if (i == len(self.NN) - 1):
				continue

			for _ in range(len(nn[0])):
				tmp.append(0.0)

			for j in range(len(nn)):
				o_k = prev_error[j] * self.sigmoid_derivative(self.res[(i + 2) * -1].dot(nn[j]) + self.bias[(i + 1) * -1][j])
				self.bias[(i + 1) * -1][j] -= sum(self.alpha * o_k)

				for k in range(len(nn[j])):
					current = self.res[(i + 2) * -1]
					w = self.alpha * o_k * current[current.columns[k]]
					nn[j][k] -= sum(w)
					tmp[k] += sum(o_k * nn[j][k])

			prev_error = tmp

