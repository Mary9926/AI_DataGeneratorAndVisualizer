import numpy as np
from window import *


class Neuron:
    epochs = 10000
    epsilon = 0.00001
    learningRate = 0.0001

    def __init__(self, activationFunction, activationFunctionDerivative):
        rng = np.random.default_rng()
        self.weights = rng.random(3)
        self.activationFunction = activationFunction
        self.activationFunctionDerivative = activationFunctionDerivative

    def neuronState(self, inputSamples):
        return inputSamples @ self.weights

    def trainNeuron(self, inputSamples, expectedOutput):
        for i in range(self.epochs):
            predictedOutput = self.learnNeuron(inputSamples)
            error = expectedOutput - predictedOutput
            error = np.mean(error)
            adjustments = self.learningRate * error * self.activationFunctionDerivative(self.neuronState(inputSamples)).reshape(
                len(inputSamples), 1) * inputSamples
            adjustments = np.mean(adjustments, 0)
            self.weights += adjustments
            if np.all(adjustments <= self.epsilon):
                break

    def learnNeuron(self, inputSamples):
        return self.activationFunction(self.neuronState(inputSamples))

    def decisionBoundary(self, xx, yy):
        xx = xx.reshape((xx.size, 1))
        yy = yy.reshape((yy.size, 1))
        inputSamples = np.c_[xx, yy, np.ones(len(xx)) * -1]
        return self.activationFunction(self.neuronState(inputSamples))


def sigmoid(s):
    b = -6
    return 1.0 / (1 + np.exp(b * s))


def sigmoidDerivative(s):
    return sigmoid(s) * (1 - sigmoid(s))


def heaviside(s):
    threshold = 0.5
    return np.heaviside(s, threshold)


def heavisideDerivative(s):
    return np.ones(s.shape)


def sin(s):
    return np.sin(s)


def sinDerivative(s):
    return np.cos(s)


def tanh(s):
    return np.tanh(s)


def tanhDerivative(s):
    return 1 - (np.tanh(s) * np.tanh(s))

