import numpy as np
from window import *


class Neuron:
    def __init__(self, activationFunction, activationFunctionDerivative):
        rng = np.random.default_rng()
        # weights of a neuron with 2 inputs
        self.weights = rng.random(2)
        self.activationFunction = activationFunction
        self.activationFunctionDerivative = activationFunctionDerivative
        print(self.weights)

    def neuronState(self, inputSamples):
        return inputSamples @ self.weights

    def trainNeuron(self, inputSamples, trainingOutput, epochs):
        for i in range(epochs):
            predictedOutput = self.learnNeuron(inputSamples)
            error = trainingOutput - predictedOutput
            # performing weight adjustments
            self.weights += error * self.activationFunctionDerivative(self.neuronState(inputSamples)) * inputSamples

    def learnNeuron(self, inputSamples):
        return self.activationFunction(neuronState(inputSamples))


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def sigmoidDerivative(y):
    return y * (1 - y)


def heaviside(s):
    return 0 if s < 0 else 1


def heavisideDerivative(s):
    return 1
