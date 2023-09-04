import layers
import numpy as np


# define our data, weights and biases
image = np.array([[1, 2, 3, 7, 5], [4, 5, 6, 7, 5], [7, 8, 9, 7, 5], [7, 8, 9, 7, 5], [7, 8, 9, 7, 5]])
kernel = np.array([[1, 2], [3, 4]])

print("\nImage:\n", image)

# instantiate layers
conv_layer = layers.ConvolutionalLayer(kernel_shape=(2, 2), stride=1)
conv_layer.setKernelWeights(kernel)
maxpool_layer = layers.MaxPoolLayer(window_shape=(2, 2), stride=1)
flatten_layer = layers.FlatteningLayer()
fully_connected_layer = layers.FullyConnectedLayer(sizeIn=9, sizeOut=5)
output_layer = layers.SquaredError()

# test the layers
forward_result = conv_layer.forward(image)
print("\nConvolutional Layer Forward Result:\n", forward_result)

forward_result = maxpool_layer.forward(forward_result)
print("\nMaxPool Layer Forward Result:\n", forward_result)

forward_result = flatten_layer.forward(forward_result)
print("\nFlattening Layer Forward Result:\n", forward_result)

forward_result = fully_connected_layer.forward(forward_result)
print("\nFully Connected Layer Forward Result:\n", forward_result)

forward_result = output_layer.eval(1, forward_result)
print("\nFully Connected Layer Forward Result:\n", forward_result)


print()