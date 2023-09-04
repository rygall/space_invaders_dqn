import gymnasium as gym
import numpy as np
import layers

env = gym.make("ALE/SpaceInvaders-v5", render_mode="human", obs_type="grayscale")

# instantiate layers
L1 = layers.ConvolutionalLayer(kernel_shape=(4, 4), stride=4)
L2 = layers.ReLuLayer()

L3 = layers.ConvolutionalLayer(kernel_shape=(2, 2), stride=2)
L4 = layers.ReLuLayer()

L5 = layers.MaxPoolLayer(window_shape=(2, 2), stride=2)
L6 = layers.FlatteningLayer()

L7 = layers.FullyConnectedLayer(sizeIn=130, sizeOut=50)
L8 = layers.ReLuLayer()

L9 = layers.FullyConnectedLayer(sizeIn=50, sizeOut=6)
L10 = layers.SqauredTemporalDifferenceError()

# assemble network
network = [L1, L2, L3, L4, L5, L6, L7, L8, L9, L10]

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