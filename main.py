#!./venv/bin/python3

import pandas as pd
import numpy as np
import sys

from MP import Multilayer_Perceptron

pd.set_option('display.float_format', lambda x: '%.8f' % x)

if __name__ == "__main__":
	df = pd.read_csv("data.csv", header=None)
	y = df[1]
	X = df.drop(1, axis=1)

	nn = Multilayer_Perceptron(X, y)
	nn.scaling()
	for i in range(100):
		print("Epoch ", i, "/100, loss - ", nn.loss(), sep='')
		nn.prop()
		nn.backprop()

	print("Success -> ", nn.test(), "%", sep='')
