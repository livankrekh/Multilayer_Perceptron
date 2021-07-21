# Multilayer_Perceptron
Pure-`numpy` version of deep learning framework (like `Keras`) for educational purposes.

## Features

* Keras-like model fitting procedure in `Multilayer_Perceptron` class
* Activation functions (Sigmoid, Tanh, ReLU, LeakyReLU, Softmax) in `Layer` class
* Additional data tools like data scaling (`Multilayer_Perceptron.data_scale()`), data splitting (`Multilayer_Perceptron.data_split()`)
* Get information about loss by `Multilayer_Perceptron.val_loss()` and `Multilayer_Perceptron.viz_loss()`
* Different Optimizers (Momentum SGD, RMSProp, Adam) and learning-rate schedulers in constructor `Multilayer_Perceptron(momentum=<bool>, nesterov=<bool>, ...)`
* Regularizations (L1, L2) in `Multilayer_Perceptron(L1=<bool>, L2=<bool> ...)`

## Usage

```python
from dl_framework.MLP import Multilayer_Perceptron, Layer

nn = Multilayer_Perceptron(X, y, layers=[Layer.tanh, Layer.tanh, Layer.tanh, Layer.softmax], n_perceptrons=[10,10,5], batch_size=5, alpha=1e-4, L1=true, L2=false, momentum=true)
nn.scaling()
nn.data_split()
nn.train(epoch=100)
nn.save("model") # to load model - use method load(<path_to_file_with_weights>)
```
