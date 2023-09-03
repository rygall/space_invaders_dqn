from abc import ABC, abstractmethod
import numpy as np
import random
np.set_printoptions(suppress=True)


class Layer(ABC):

    def __init__(self):
        self.__prevIn = []
        self.__prevOut = []

    def setPrevIn(self, dataIn):
        self.__prevIn = dataIn

    def setPrevOut(self, out):
        self.__prevOut = out

    def getPrevIn(self):
        return self.__prevIn

    def getPrevOut (self):
        return self.__prevOut

    @abstractmethod
    def forward(self, dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass


class InputLayer(Layer):

    def __init__(self, dataIn):
        super().__init__()
        self.mean = np.mean(dataIn, axis=0)
        self.std = np.std(dataIn, axis=0, ddof=1)
        if isinstance(self.std, float):
            if self.std == 0:
                self.std = 1
        else:
            for i in range(0, len(self.std)):
                if self.std[i] == 0:
                    self.std[i] = 1

    def forward(self, dataIn):
        self.setPrevIn = dataIn
        zscored_matrix = (dataIn - self.mean) / self.std
        self.setPrevOut = zscored_matrix
        return zscored_matrix

    def gradient(self):
        pass

    def backward (self, gradIn):
        pass


class FullyConnectedLayer(Layer):

    def __init__(self, sizeIn, sizeOut, eta=0.0001, weights='base', lr_method='base'):
        super().__init__()
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        self.weights, self.biases = self.initWeights(sizeIn, sizeOut, weights)
        self.eta = eta
        self.lr_method = lr_method
        if self.lr_method == 'adam':
            self.s = 0
            self.r = 0

    def initWeights(self, sizeIn, sizeOut, method):
        if method == 'base':
            weights = np.random.uniform(low=-0.0001, high=0.0001, size=(sizeIn, sizeOut))
            biases = np.random.uniform(low=-0.0001, high=0.0001, size=sizeOut)
            return weights, biases
        if method == 'xavier':
            weights = np.random.uniform(low=-np.sqrt(6/(self.sizeIn+self.sizeOut)), high=-np.sqrt(6/(self.sizeIn+self.sizeOut)), size=(sizeIn, sizeOut))
            biases = np.zeros(self.sizeOut, dtype=float)
            return weights, biases
        
    def getWeights(self):
        return self.weights

    def getBiases(self):
        return self.biases

    def setWeights(self, weights):
        new_shape = np.shape(weights)
        req_shape = np.shape(self.weights)
        if new_shape == req_shape:
            self.weights = np.array(weights, dtype=float) 
        else:
            print("mismatch in shape of input weights to required shape of weights: input_shape =", new_shape, "required_shape =", req_shape)

    def setBiases(self, biases):
        new_shape = np.shape(biases)
        req_shape = np.shape(self.biases)
        if new_shape == req_shape: 
            self.biases = np.array(biases, dtype=float)
        else:
            print("mismatch in shape of input biases to required shape of biases: input_shape =", new_shape, "required_shape =", req_shape)

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = dataIn @ self.weights + self.biases
        self.setPrevOut(y)
        return y

    def gradient(self):
        return np.transpose(self.weights)   

    def updateWeights(self, gradIn, epoch):
        dJdb = np.sum(gradIn, axis=0) / gradIn.shape[0]
        x = self.getPrevIn().T
        dJdW = (self.getPrevIn().T @ gradIn) / gradIn.shape[0]
        self.biases -= self.eta*dJdb
        if self.lr_method == 'base':
            self.weights -= self.eta*dJdW
        if self.lr_method == 'adam':
            rho_1 = 0.9
            rho_2 = 0.999
            delta = 10**-8
            self.s = (rho_1*self.s) + ((1-rho_1)*dJdW)
            self.r = (rho_2*self.r) + ((1-rho_2)*(dJdW*dJdW))
            # update weight based on gradient
            z = (self.s/(1-(rho_1**epoch))) / (np.sqrt(self.r/(1-(rho_2**epoch))) + delta)
            self.weights = self.weights - (self.eta*z)

    def backward (self, gradIn):
        gradOut = gradIn @ self.gradient()
        return gradOut


class LinearLayer(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(dataIn) 
        return dataIn

    def gradient(self):
        a = np.ones(np.shape(self.getPrevOut()))
        if a.ndim == 1:
            gradient = np.eye(len(a)) * a
        else:
            gradient = np.eye(np.size(self.getPrevOut(), axis=1)) * a[:, np.newaxis, :]
        return gradient
    
    def backward (self, gradIn):
        delta = np.array(gradIn)
        dgdz = self.gradient()
        if dgdz.ndim == 3:
            gradOut = np.einsum('...i, ...ij', delta, dgdz)
        else:
            gradOut = delta * dgdz
        return gradOut


class ReLuLayer(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = np.maximum(0.0, dataIn)
        self.setPrevOut(y) 
        return y
    
    def gradient(self):
        a = np.where(self.getPrevOut() < 0, 0, 1)
        if a.ndim == 1:
            gradient = np.eye(len(a)) * a
        else:
            gradient = np.eye(np.size(self.getPrevOut(), axis=1)) * a[:, np.newaxis, :]
        return gradient
    
    def backward (self, gradIn):
        delta = np.array(gradIn)
        dgdz = self.gradient()
        if (dgdz.ndim == 3):
            gradOut = np.einsum('...i, ...ij', delta, dgdz)
        else:
            gradOut = delta * dgdz
        return gradOut


class LogisticSigmoidLayer(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = 1 / (1+np.exp(-dataIn))
        self.setPrevOut(y)
        return y

    def gradient(self):
        a = self.getPrevOut() * (1 - self.getPrevOut())
        if a.ndim == 1:
            gradient = np.eye(len(a)) * a
        else:
            gradient = np.eye(np.size(self.getPrevOut(), axis=1)) * a[:, np.newaxis, :]
        return gradient
    
    def backward (self, gradIn):
        delta = np.array(gradIn)
        dgdz = self.gradient()
        if dgdz.ndim == 3:
            gradOut = np.einsum('...i, ...ij', delta, dgdz)
        else:
            gradOut = delta * dgdz
        return gradOut


class TanhLayer(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = (np.exp(dataIn) - np.exp(-dataIn)) / (np.exp(dataIn) + np.exp(-dataIn))
        self.setPrevOut(y) 
        return y

    def gradient(self):
        a = 1 - (np.power(self.getPrevOut(), 2))
        if a.ndim == 1:
            gradient = np.eye(len(a)) * a
        else:
            gradient = np.eye(np.size(self.getPrevOut(), axis=1)) * a[:, np.newaxis, :]
        return gradient
    
    def backward (self, gradIn):
        delta = np.array(gradIn)
        dgdz = self.gradient()
        if dgdz.ndim == 3:
            gradOut = np.einsum('...i, ...ij', delta, dgdz)
        else:
            gradOut = delta * dgdz
        return gradOut


class SoftmaxLayer(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        a = dataIn
        if a.ndim == 1:
            max_value = np.max(dataIn)
            exp = np.exp(dataIn - max_value)
            sum = np.sum(exp)
            y = exp / sum
        else:
            max_values = np.max(dataIn, axis=1)
            max_values = np.atleast_2d(max_values).T
            exp = np.exp(dataIn - max_values)
            sum = np.atleast_2d(np.sum(exp, axis=1)).T
            y = exp / sum
        self.setPrevOut(y) 
        return y

    def gradient(self):
        x = self.getPrevOut()
        if x.ndim == 1:
            diag = np.eye(len(x)) * x
            a = np.atleast_2d(x).T * x
            gradient = diag - a
        else:
            diag = np.eye(np.size(x, axis=1)) * x[:, np.newaxis, :]
            a = []
            for i in range(0, np.size(x, axis=0)):
                result = np.atleast_2d(x[i]).T * x[i]
                a.append(result)
            gradient = diag - a
        return gradient

    def backward(self, gradIn):
        delta = np.array(gradIn)
        dgdz = self.gradient()
        if dgdz.ndim == 3:
            gradOut = np.einsum('...i, ...ij', delta, dgdz)
        else:
            gradOut = delta * dgdz
        return gradOut


class Dropout(Layer):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, dataIn, mode='inactive'):
        if mode == 'active':
            self.setPrevIn(dataIn)
            p = np.random.uniform(low=0.0, high=1.0, size=np.size(dataIn, axis=1))
            gz = dataIn
            for i in range(0, np.size(dataIn, axis=1)):
                if p[i] < self.p:
                    gz[:, i] = 0.0
                else:
                    gz[:, i]*(1/(1-self.p))
            gz = np.array(gz)
            self.setPrevOut(gz) 
            return gz
        else:
            self.setPrevIn(dataIn)
            self.setPrevOut(dataIn)
            return dataIn

    def gradient(self):
        a = self.getPrevOut()
        for i in range(0, np.size(a, axis=1)):
            a[:, i] = (1/(1-self.p)) * (a[:, i] == 0.0)
        if a.ndim == 1:
            gradient = np.eye(len(a)) * a
        else:
            gradient = np.eye(np.size(self.getPrevOut(), axis=1)) * a[:, np.newaxis, :]
        return gradient
    
    def backward (self, gradIn):
        delta = np.array(gradIn)
        dgdz = self.gradient()
        if (dgdz.ndim == 3):
            gradOut = np.einsum('...i, ...ij', delta, dgdz)
        else:
            gradOut = delta * dgdz
        return gradOut


class SquaredError():

    def eval(self, Y, Yhat):
        a = np.mean((Y - Yhat) * (Y - Yhat))
        return a

    def gradient(self, Y, Yhat):
        a = -2*(Y-Yhat)
        return a


class LogLoss():

    def eval(self, Y, Yhat):
        epsilon = 0.0000001
        a = np.mean(-((Y * np.log(Yhat+epsilon)) + ((1-Y) * np.log(1-Yhat+epsilon))))
        return a

    def gradient(self, Y, Yhat):
        epsilon = 0.0000001
        a = -(Y-Yhat) / (Yhat * (1 - Yhat) + epsilon)
        return a


class CrossEntropy():

    def eval(self, Y, Yhat):
        epsilon = 0.0000001
        x = np.array(Y)
        if x.ndim == 1:
            a = -np.mean(np.sum((Y * np.log(Yhat+epsilon)), axis=0))
        else:
            a = -np.mean(np.sum((Y * np.log(Yhat+epsilon)), axis=1))
        return a

    def gradient(self, Y, Yhat):
        epsilon = 0.0000001
        a = -Y / (Yhat + epsilon)
        return a