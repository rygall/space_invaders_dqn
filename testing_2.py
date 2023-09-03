import layers
import numpy as np
import matplotlib.pyplot as plt


# define our data, weights and biases
image = np.array([[1, 2, 3, 7, 5], [4, 5, 6, 7, 5], [7, 8, 9, 7, 5], [7, 8, 9, 7, 5], [7, 8, 9, 7, 5]])
kernel = np.array([[1, 2], [3, 4]])
print("\nImage:\n", image)

# instantiate layers
L1 = layers.ConvolutionalLayer(kernel_shape=(2, 2), stride=1)
L1.setKernelWeights(kernel)
L2 = layers.MaxPoolLayer(window_shape=(2, 2), stride=1)
L3 = layers.FlatteningLayer()

L4 = layers.FullyConnectedLayer(sizeIn=9, sizeOut=5)
L5 = layers.LinearLayer()

L6 = layers.SquaredError()

# assemble network
network = [L1, L2, L3, L4, L5, L6]

# define training parameters
max_epochs = 300

# gradient descent
for epoch in range(1, max_epochs):

    # forward propogation
    t = image
    for k in range(len(network)-1):
        t = network[k].forward(t)

    # backward propogation
    grad = network[-1].gradient(1, t)
    for z in range(len(network)-2, 0, -1):
        newgrad = network[z].backward(grad)
        #if(isinstance(network[z], layers.FullyConnectedLayer)):
            #network[z].updateWeights(np.array(grad), epoch)
        grad = newgrad