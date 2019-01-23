#!./venv/bin/python3

import pandas as pd
import numpy as np
import sys

from MP import Multilayer_Perceptron, Layer

pd.set_option('display.float_format', lambda x: '%.8f' % x)
np.random.seed(7483)

if __name__ == "__main__":
	df = pd.read_csv("data.csv", header=None)
	y = df[1]
	X = df.drop(1, axis=1)

	nn = Multilayer_Perceptron(X, y, layers=[Layer.tanh, Layer.tanh, Layer.tanh, Layer.tanh, Layer.softmax], n_perceptrons=[10,10,5], batch_size=5)
	nn.scaling()
	nn.train()
	print("Success -> ", nn.test(), "%", sep='')

	nn.viz_loss()
