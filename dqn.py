import numpy as np
import layers
import random

class DQN():

    def __init__(self, epsilon=0.75):
        self.network = self.network()
        self.target_network = self.network
        self.epsilon = epsilon
        self.prev_action = None
        self.prev_state = None
        self.prev_q = None

    def network(self):
        # instantiate layers
        L0 = layers.InputLayer(np.zeros((210, 160)))
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
        network = [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10]
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
        return self.prev_action

    def train(self, state, reward, epoch):

        # forward propogation through target network
        h = state
        for i in range(len(self.target_network)-1):
            h = self.target_network[i].forward(h)

        # backward propogation
        grad = self.network[-1].gradient(self.prev_action, self.prev_q, h, reward)
        for z in range(len(self.network)-2, 0, -1):
            newgrad = self.network[z].backward(grad)
            if(isinstance(self.network[z], layers.FullyConnectedLayer)):
                self.network[z].updateWeights(np.array(grad), epoch)
            if(isinstance(self.network[z], layers.ConvolutionalLayer)):
                self.network[z].updateWeights(np.array(grad))
            grad = newgrad

    def updateTarget(self):
        self.target_network = self.network

    def save(self):
        np.save("L2.npy", self.network[1].getKernel())
        np.save("L4.npy", self.network[3].getKernel())
        np.save("L8.npy", self.network[7].getWeights())
        np.save("L8_bias.npy", self.network[7].getBiases())
        np.save("L10.npy", self.network[9].getWeights())
        np.save("L10_bias.npy", self.network[9].getBiases())

    def load(self):
        self.network[1].setKernel(np.load("L2.npy"))
        self.network[3].setKernel(np.load("L4.npy"))
        self.network[7].setWeights(np.load("L8.npy"))
        self.network[7].setBiases(np.load("L8_bias.npy"))
        self.network[9].setWeights(np.load("L10.npy"))
        self.network[9].setBiases(np.load("L10_bias.npy"))
