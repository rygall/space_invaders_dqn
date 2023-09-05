from abc import ABC, abstractmethod
import numpy as np
import math
from numba import jit
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

    def __init__(self, dataIn, zscore='False'):
        super().__init__()
        self.zscore = zscore
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
        if self.zscore != 'False': 
            zscored_matrix = (dataIn - self.mean) / self.std
            self.setPrevOut = zscored_matrix
            return zscored_matrix
        else:
            self.setPrevOut(dataIn)
            return dataIn
        
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

    @jit
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = dataIn @ self.weights + self.biases
        self.setPrevOut(y)
        return y

    def gradient(self):
        return np.transpose(self.weights)   

    @jit
    def updateWeights(self, gradIn, epoch):
        dJdb = np.sum(gradIn, axis=0) / gradIn.shape[0]
        x = self.getPrevIn()
        x_T = self.getPrevIn().T
        x_T_2 = np.atleast_2d(self.getPrevIn()).T
        dJdW = (x_T_2 * gradIn) / gradIn.shape[0]
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

    @jit
    def backward (self, gradIn):
        gradOut = gradIn @ self.gradient()
        return gradOut


class ReLuLayer(Layer):

    def __init__(self):
        super().__init__()

    @jit
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = np.maximum(0.0, dataIn)
        self.setPrevOut(y) 
        return y
    
    @jit
    def gradient(self):
        a = np.where(self.getPrevOut() < 0, 0, 1)
        if a.ndim == 1:
            gradient = np.eye(len(a)) * a
        else:
            gradient = np.eye(np.size(self.getPrevOut(), axis=1)) * a[:, np.newaxis, :]
        return gradient
    
    @jit
    def backward (self, gradIn):
        delta = np.array(gradIn)
        dgdz = self.gradient()
        gradOut = np.einsum('...i, ...ij', delta, dgdz)
        return gradOut


class ConvolutionalLayer(Layer):

    def __init__(self, kernel_shape, stride=1, eta=0.0001):
        super().__init__()
        self.kernel = np.random.rand(kernel_shape[0], kernel_shape[1])
        self.stride = stride
        self.eta = eta

    def setKernel(self, kernel):
        self.kernel = kernel

    def getKernel(self):
        return self.kernel

    @jit
    def updateWeights(self, gradIn):
        dJdK = self.grad_correlate(gradIn)
        self.kernel -= self.eta*dJdK

    @jit
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = self.correlate(dataIn)
        self.setPrevOut(dataOut)
        return dataOut

    @jit
    def correlate(self, image):
        image_x, image_y = image.shape
        kernel_x, kernel_y = self.kernel.shape
        feature_x = 1+int((image_x-kernel_x)/self.stride)
        feature_y = 1+int((image_y-kernel_y)/self.stride)
        feature_map = np.zeros((feature_x, feature_y))
        for a in range(feature_x):
            for b in range(feature_y):
                sum = 0
                for i in range(1, kernel_x+1):
                    for j in range(1, kernel_y+1):
                        pos_1 = int(a - (kernel_x//2) + i)
                        pos_2 = int(b - (kernel_y//2) + j)
                        sum += image[pos_1][pos_2] * self.kernel[(i-1)][(j-1)] 
                feature_map[a, b] = sum
        return feature_map
    
    @jit
    def grad_correlate(self, gradIn):
        prevIn_x, prevIn_y = self.getPrevIn().shape
        gradIn_x, gradIn_y = gradIn.shape
        feature_x = 1+int((prevIn_x-gradIn_x)/self.stride)
        feature_y = 1+int((prevIn_y-gradIn_y)/self.stride)
        gradient = np.zeros((feature_x, feature_y))
        for a in range(feature_x):
            for b in range(feature_y):
                sum = 0
                for i in range(1, gradIn_x+1):
                    for j in range(1, gradIn_y+1):
                        pos_1 = int(a - (gradIn_x//2) + i)
                        pos_2 = int(b - (gradIn_y//2) + j)
                        sum += self.getPrevIn()[pos_1][pos_2] * gradIn[(i-1)][(j-1)] 
                gradient[a, b] = sum
        return gradient
    
    @jit
    def backprop_correlate(self, pad_grad):
        pad_grad_x, pad_grad_y = pad_grad.shape
        kernel_transpose = np.transpose(self.kernel)
        kt_x, kt_y = kernel_transpose.shape
        feature_x = 1+int((pad_grad_x-kt_x)/self.stride)
        feature_y = 1+int((pad_grad_y-kt_y)/self.stride)
        gradOut = np.zeros((feature_x, feature_y))
        for a in range(feature_x):
            for b in range(feature_y):
                sum = 0
                for i in range(1, kt_x+1):
                    for j in range(1, kt_y+1):
                        pos_1 = int(a - (kt_x//2) + i)
                        pos_2 = int(b - (kt_y//2) + j)
                        sum += pad_grad[pos_1][pos_2] * kernel_transpose[(i-1)][(j-1)] 
                gradOut[a, b] = sum
        return gradOut
    
    @jit
    def gradient(self, gradIn):
        gradient = self.grad_correlate(gradIn)
        return gradient
    
    @jit
    def backward(self, gradIn):
        kernel_x = self.kernel.shape[0]  
        m = math.ceil(kernel_x / 2) + 1
        djdf = np.array(gradIn)
        djdf_padded = np.pad(djdf, ((m, m), (m, m)), mode='constant', constant_values=0)
        gradOut = self.backprop_correlate(djdf_padded)
        return gradOut
            

class MaxPoolLayer(Layer):
    
    def __init__(self, window_shape, stride=1):
        super().__init__()
        self.window_shape = window_shape
        self.stride = stride

    def setKernelWeights(self, weights):
        self.kernel = weights

    def getKernel(self):
        return self.kernel

    @jit
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = self.pool(dataIn)
        self.setPrevOut(dataOut)
        return dataOut

    @jit
    def pool(self, map):
        map_x, map_y = map.shape
        window_x = self.window_shape[0]
        window_y = self.window_shape[1]
        feature_x = 1+int((map_x-window_x)/self.stride)
        feature_y = 1+int((map_y-window_y)/self.stride)
        feature_map = np.zeros((feature_x, feature_y))
        for a in range(feature_x):
            for b in range(feature_y):
                max = float('-inf')
                for i in range(1, window_x+1):
                    for j in range(1, window_y+1):
                        pos_1 = int(a - (window_x//2) + i)
                        pos_2 = int(b - (window_y//2) + j)
                        if map[pos_1][pos_2] > max:
                            max = map[pos_1][pos_2]
                feature_map[a, b] = max
        return feature_map
    
    @jit
    def map(self, prev_in, gradIn):
        prev_x, prev_y = prev_in.shape
        grad = np.zeros((prev_x, prev_y))
        window_x = self.window_shape[0]
        window_y = self.window_shape[1]
        slides_x = 1+int((prev_x-window_x)/self.stride)
        slides_y = 1+int((prev_y-window_y)/self.stride)
        for a in range(slides_x):
            for b in range(slides_y):
                index_a = a * self.stride
                index_b = b * self.stride
                max = float('-inf')
                max_index = [0, 0]
                for i in range(1, window_x+1):
                    for j in range(1, window_y+1):
                        pos_1 = int(index_a - (window_x//2) + i)
                        pos_2 = int(index_b - (window_y//2) + j)
                        if prev_in[pos_1][pos_2] > max:
                            max = prev_in[pos_1][pos_2]
                            max_index[0] = pos_1
                            max_index[1] = pos_2
                grad[max_index[0], max_index[1]] = gradIn[a, b]
        return grad
    
    @jit
    def gradient(self, gradIn):
        prev_in = self.getPrevIn()
        grad = self.map(prev_in, gradIn)
        return grad

    @jit
    def backward (self, gradIn):
        gradOut = self.gradient(gradIn)
        return gradOut


class FlatteningLayer(Layer):
    
    def __init__(self):
        super().__init__()

    @jit
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = dataIn.flatten()
        self.setPrevOut(dataOut)
        return dataOut
    
    @jit
    def gradient(self, gradIn):
        gradient = np.reshape(gradIn, self.getPrevIn().shape, order='C')
        return gradient
    
    @jit
    def backward (self, gradIn):
        gradOut = self.gradient(gradIn)
        return gradOut


class SquaredTemporalDifferenceError():

    def __init__(self, gamma=0.5):
        self.gamma = gamma
    
    @jit
    def gradient(self, action, Q, Q_next, reward):
        q_next_max = Q_next.max()
        q_new = (Q - (reward + (self.gamma*q_next_max)))**2
        grad = np.zeros(q_new.shape)
        grad[action] = q_new[action] 
        return grad