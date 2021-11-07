import numpy as np
from window import *


class Neuron:
    epochs = 1000
    epsilon = 0.001

    def __init__(self, activationFunction, activationFunctionDerivative):
        rng = np.random.default_rng()
        # weights of a neuron with 3 inputs
        self.weights = rng.random(3)
        self.activationFunction = activationFunction
        self.activationFunctionDerivative = activationFunctionDerivative

    def neuronState(self, inputSamples):
        return inputSamples @ self.weights

    def trainNeuron(self, inputSamples):
        print("Input Samples: ")
        print(inputSamples)
        print("Init Weights: ")
        print(self.weights)
        print("Expected Output: ")
        expectedOutput = self.weights[:, np.newaxis].T @ inputSamples.T
        print(expectedOutput)
        for i in range(self.epochs):
            predictedOutput = self.learnNeuron(inputSamples)
            print("Output: ")
            print(predictedOutput.shape)
            print(predictedOutput)
            error = expectedOutput - predictedOutput
            print("Error: ")
            print(error.shape)
            print(error)
            print("To: ")
            print((self.activationFunctionDerivative(self.neuronState(inputSamples))).shape)
            print(self.activationFunctionDerivative(self.neuronState(inputSamples)))
            adjustments = error * self.activationFunctionDerivative(self.neuronState(inputSamples)) * inputSamples
            self.weights += adjustments
            if np.all(adjustments <= epsilon):
                break

    def learnNeuron(self, inputSamples):
        return self.activationFunction(self.neuronState(inputSamples))


def sigmoid(s):
    b = -6
    return 1 / (1 + np.exp(b * s))


def sigmoidDerivative(y):
    return y * (1 - y)


def heaviside(s):
    return 0 if s < 0 else 1


def heavisideDerivative(s):
    return 1
