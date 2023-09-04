import gymnasium as gym
import numpy as np
import layers

env = gym.make("ALE/SpaceInvaders-v5", render_mode="human", obs_type="grayscale")
kernel = np.array([[1, 2], [3, 4]])

L1 = layers.ConvolutionalLayer(kernel_shape=(2, 2), stride=1)
L1.setKernelWeights(kernel)
L2 = layers.MaxPoolLayer(window_shape=(2, 2), stride=1)
L3 = layers.FlatteningLayer()

L4 = layers.FullyConnectedLayer(sizeIn=32864, sizeOut=6)
L5 = layers.LinearLayer()
L6 = layers.SqauredTemporalDifferenceError()

# assemble network
network = [L1, L2, L3, L4, L5, L6]

# define training parameters
max_epochs = 300
observation, info = env.reset()

action = 1
    
for epoch in range(1, max_epochs):
    observation, reward, terminated, truncated, info = env.step(action)
    
    # forward propogation
    t = observation
    for k in range(len(network)-1):
        t = network[k].forward(t)
    
    # backward propogation
    grad = network[-1].gradient([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], t, 5)
    for z in range(len(network)-2, 0, -1):
        newgrad = network[z].backward(grad)
        #if(isinstance(network[z], layers.FullyConnectedLayer)):
            #network[z].updateWeights(np.array(grad), epoch)
        grad = newgrad
        
    print("grad", grad.shape)
    if terminated or truncated:
        observation, info = env.reset()
        
env.close()