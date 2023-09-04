import numpy as np
import layers

class DQN():

    def __init__(self):
        self.network = self.getNetwork()
        self.target_network = self.network

    def getNetwork(self):
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
        L10 = layers.SquaredTemporalDifferenceError()
        # assemble network
        network = [L1, L2, L3, L4, L5, L6, L7, L8, L9, L10]
        return network

    def getAction(self, env):
        # forward propogation through network
        t = env
        for k in range(len(self.network)-1):
            t = self.network[k].forward(t)
        # return the index of max Q-value
        return t.argmax()   

    def train(self, dataIn):
        # forward propogation through network
        t = dataIn
        for k in range(len(self.network)-1):
            t = self.network[k].forward(t)

        # forward propogation through target network
        h = dataIn
        for i in range(len(self.target_network)-1):
            h = self.target_network[i].forward(h)
    
        # backward propogation
        grad = self.network[-1].gradient(h, t, 5)
        for z in range(len(self.network)-2, 0, -1):
            newgrad = self.network[z].backward(grad)
            #if(isinstance(network[z], layers.FullyConnectedLayer)):
                #network[z].updateWeights(np.array(grad), epoch)
            #if(isinstance(network[z], layers.ConvolutionalLayer)):
                #network[z].updateWeights(np.array(grad), epoch)
            grad = newgrad

    def copy(self):
        pass

    def save(self):
        pass