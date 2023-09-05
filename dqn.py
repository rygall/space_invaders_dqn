import numpy as np
import layers
import random

class DQN():

    def __init__(self, epsilon=0.75):
        self.network = self.getNetwork()
        self.target_network = self.network
        self.epsilon = epsilon
        self.prev_action = None
        self.prev_state = None
        self.prev_q = None

    def getNetwork(self):
        # instantiate layers
        L1 = layers.ConvolutionalLayer(kernel_shape=(4, 4))
        L2 = layers.ReLuLayer()
        L3 = layers.ConvolutionalLayer(kernel_shape=(4, 4))
        L4 = layers.ReLuLayer()
        L5 = layers.MaxPoolLayer(window_shape=(4, 4), stride=4)
        L6 = layers.FlatteningLayer()
        L7 = layers.FullyConnectedLayer(sizeIn=1938, sizeOut=100)
        L8 = layers.ReLuLayer()
        L9 = layers.FullyConnectedLayer(sizeIn=100, sizeOut=6)
        L10 = layers.SquaredTemporalDifferenceError()
        # assemble network
        network = [L1, L2, L3, L4, L5, L6, L7, L8, L9, L10]
        return network

    def action(self, state):

        # forward propogation through network
        t = state
        for k in range(len(self.network)-1):
            t = self.network[k].forward(t)

        #store state, action, and associated q value
        self.prev_state = state
        rand = random.uniform(0, 1)
        if rand > self.epsilon:
            self.prev_action = random.randint(0, 5)
        else:
            self.prev_action = t.argmax()
        self.prev_q = t
        return t.argmax()

    def train(self, state, reward, epoch):

        # forward propogation through target network
        h = state
        for i in range(len(self.target_network)-1):
            h = self.target_network[i].forward(h)

        # backward propogation
        grad = self.network[-1].gradient(self.prev_action, self.prev_q, h, reward)
        for z in range(len(self.network)-2, 0, -1):
            newgrad = self.network[z].backward(grad)
            #if(isinstance(self.network[z], layers.FullyConnectedLayer)):
                #self.network[z].updateWeights(np.array(grad), epoch)
            #if(isinstance(self.network[z], layers.ConvolutionalLayer)):
                #self.network[z].updateWeights(np.array(grad))
            grad = newgrad

    def copy(self):
        pass

    def save(self):
        pass