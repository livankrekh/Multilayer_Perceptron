#!./venv/bin/python3

import pandas as pd
import numpy as np
import argparse
import sys

from MP import Multilayer_Perceptron, Layer

pd.set_option('display.float_format', lambda x: '%.8f' % x)
np.random.seed(7483)

if __name__ == "__main__":

	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--epochs", required = False, help = "Number of epochs")
	ap.add_argument("-a", "--alpha", required = False, help = "Training speed")
	ap.add_argument("-l1", "--l1", required = False, help = "L1 normalization koff")
	ap.add_argument("-l2", "--l2", required = False, help = "L2 normalization koff")
	ap.add_argument("-n", "--nesterov", required = False, action="store_true", help = "Nesterov momentum optimization")
	ap.add_argument("-f", "--file", required = True, help = "Path to dataset")
	args = vars(ap.parse_args())

	df = pd.read_csv(args['file'], header=None)
	y = df[1]
	X = df.drop(1, axis=1)

	n_epochs = ( 200 if args['epochs'] == None else int(args['epochs']) )
	alpha = (0.04 if args['alpha'] == None else float(args['alpha']) )
	l1 = (0.0 if args['l1'] == None else float(args['l1']))
	l2 = (0.0 if args['l2'] == None else float(args['l2']))

	nn = Multilayer_Perceptron(X, y, layers=[Layer.tanh, Layer.tanh, Layer.tanh, Layer.softmax], n_perceptrons=[10,10,5], batch_size=5, alpha=alpha, L1=l1, L2=l2, nesterov=args['nesterov'])
	nn.scaling()
	nn.data_split()
	nn.train(epoch=n_epochs)
	print("\033[1m\033[32mSuccess -> ", nn.test(), "%\033[0m", sep='')

	nn.viz_loss()
	nn.save("model")
