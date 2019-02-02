#!./venv/bin/python3

import pandas as pd
import numpy as np

import sys

from MP import Multilayer_Perceptron, Layer

if __name__ == "__main__":
	df = pd.read_csv(sys.argv[1], header=None)
	df = df.drop(columns=1)

	for i in df.columns:
		if df[i].dropna().empty:
			df = df.drop(columns=i)

	nn = Multilayer_Perceptron(df)
	nn.loadNN('model.npy')
	nn.scaling()
	
	predicts = nn.predict()
	predict_df = pd.DataFrame({'Index':range(len(predicts)), 'Predicts':predicts})
	predict_df.to_csv('predicts.csv', index=False)

	print(predict_df)
	print("\033[1m\033[32mCross validation -", nn.cross_val(predicts), "\033[0m")
	print("\033[1m\033[32mResults saved to predicts.csv\033[0m")
