import numpy as np


class ExperienceReplay():
    def __init__(self, size = 50):
        self.observations = []
        self.size = size
        pass
    
    def getObs(self):
        return self.observations
    
    def addObs(self, observation, reward, action):
        if len(self.observations) == self.size:
            self.observations.pop(0)
            
        self.observations.append([observation, reward, action])
        
    def getSamples(self, numSamples):
        if numSamples >= len(self.observations):
            return self.observations
        else:
            indeces = np.random.choice(len(self.observations), numSamples, replace = False)
            returnObs = []
            for i in indeces:
                returnObs.append(self.observations[i])
            
            return returnObs