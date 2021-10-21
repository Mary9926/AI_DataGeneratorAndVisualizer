import numpy as np
from window import *


class Neuron():
    def __init__(self):
        ng = np.random.default_rng()
        self.weights = np.random.random(-1, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        return x * (1 - x)

    def trainNeuron(self, trainingInputs, trainingOutputs, trainingIterations):
        for i in range(trainingIterations):
            output = self.learn(trainingInputs)
            error = trainingOutputs - output
            # performing weight adjustments
            adjustments = np.dot(trainingInputs.T, error * self.sigmoidDerivative(output))
            self.weights += adjustments

    def learn(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.weights))
        return output


