Intent is to create and run neural network using GPU.

First need to [test that GPU works](http://deeplearning.net/software/theano/tutorial/using_gpu.html) ([test_gpu.py](test_gpu.py)).

## Convoluted Neural Network (CNN)

Then want to construct a CNN ([test_cnn.py](test_cnn.py)).

CNN structure might look like:

```
- Input layer
- Convolutional layer (ReLU is the common nonlinearity in CNNs)
- Pooling layer
- ... You can add more convolutional & pooling layers ...
- Output layer
```

You will also need to define the learning during training e.g.

- What is the objective / loss function
- how the weights are updated
