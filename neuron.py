import numpy as np
from window import *


class Neuron:
    epochs = 10000
    epsilon = 0.00001

    def __init__(self, activationFunction, activationFunctionDerivative):
        rng = np.random.default_rng()
        # weights of a neuron with 3 inputs
        self.weights = rng.random(3)
        self.activationFunction = activationFunction
        self.activationFunctionDerivative = activationFunctionDerivative

    def neuronState(self, inputSamples):
        return inputSamples @ self.weights

    def trainNeuron(self, inputSamples, expectedOutput):
        # print("Input Samples: ")
        # print(inputSamples)
        # print("Init Weights: ")
        # print(self.weights)
        # print("Expected Output: ")
        # print(expectedOutput)
        for i in range(self.epochs):
            predictedOutput = self.learnNeuron(inputSamples)
            # print("Output: ")
            # print(predictedOutput.shape)
            # print(predictedOutput)
            error = expectedOutput - predictedOutput
            # print("Error: ")
            # print(error.shape)
            # print(error)
            # print("To: ")
            # print(self.activationFunctionDerivative(self.neuronState(inputSamples)).reshape(len(inputSamples), 1).shape)
            # print(self.activationFunctionDerivative(self.neuronState(inputSamples)).reshape(len(inputSamples), 1))
            error = np.mean(error)
            adjustments = error * self.activationFunctionDerivative(self.neuronState(inputSamples)).reshape(len(inputSamples), 1) * inputSamples
            # print("Adjustments: ")
            # print(adjustments)
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


