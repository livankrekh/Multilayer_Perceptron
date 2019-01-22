import pandas as pd
import numpy as np

# np.random.seed(67)
np.random.seed(8954)

class Multilayer_Perceptron(object):
	def __init__(self, X, y, n_perceptrons=[20,20], batch_size=20, validator=False):
		self.X = X
		self.y = y
		self.y_all = y
		self.y_name = np.unique(self.y)
		self.n_p = n_perceptrons
		self.res = []
		self.bias = []
		self.alpha = 0.04
		self.NN = []
		self.output_layer = []
		self.b_s = batch_size
		self.valid = validator

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

	def test(self):
		output = np.array(self.predict())
		y = np.array(self.y_all)

		for i in range(len(y)):
			output[i] = self.y_name.tolist().index(output[i])
			y[i] = self.y_name.tolist().index(y[i])

		res = output.astype(int) - y.astype(int)
		res = np.where(res == 0, 1, 0)

		return int(sum(res) / len(res) * 100)

	def predict(self):
		self.res = []
		self.res.append(self.X)

		self.res[-1] = pd.DataFrame(self.res[-1].as_matrix())

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

		self.res[-1] = pd.DataFrame(self.res[-1].as_matrix())

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

			prev_error.append(self.res[-1][i] - pd.Series(y))

		for i, nn in enumerate(reversed(self.NN)):
			tmp = []

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

