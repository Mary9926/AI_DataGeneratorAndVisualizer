import numpy as np
from window import *


class Neuron:
    epochs = 10000
    epsilon = 0.00001
    learningRate = 0.1

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


class NeuralNetwork:
    epochs = 1#10000
    eta = 0.1

    def __init__(self):
        neuronAmount = 2
        inputAmount = 2
        self.layers = [Linear(neuronAmount, inputAmount), Activation,
                       Linear(neuronAmount, inputAmount), Activation,
                       Linear(neuronAmount, inputAmount), Activation]


    def trainNeuralNetwork(self, inputSamples, expectedOutput):
        for i in range(self.epochs):
            #predictedOutput = Linear.forwardPropagation(self, inputSamples)
            gradient = np.mean((expectedOutput - predictedOutput) * (expectedOutput - predictedOutput))
            for layer in self.layers[::-1]: #reverse order
                gradient = layer.backPropagation(gradient)
            for layer in self.layers:
                predictedOutput = layer.forwardPropagation(predictedOutput)
                layer.adjust(eta)

    def adjustments(self):
        for layer in self.layers:
            layer.adjust(eta)


class Linear:
    
    def __init__(self, neuronAmount, inputAmount):
        rng = np.random.default_rng()
        self.weights = []
        self.inputAmount = inputAmount
        self.neuronAmount = neuronAmount

        for i in range(0, neuronAmount):
            w = rng.random(3)
            self.weights.append(w)
        self.weights = np.array(self.weights)

    def forwardPropagation(self, inputSamples):
        self.inputSamples = inputSamples
        return self.inputSamples @ self.weights.T

    def backPropagation(self, gradient):
        self.gradient = gradient
        print('gradient')
        print(gradient)
        print('gradient @ weights')
        print(self.gradient @ self.weights)
        return self.gradient @ self.weights

    def adjust(self, eta):
        print('weights')
        self.weights += eta * self.gradient * self.inputSamples
        print(eta * self.gradient * self.inputSamples)


class Activation:

    def __init__(self):
        self.state = state

    def forwardPropagation(self, state):
        return self.sigmoid(state)

    def backPropagation(self, gradient):
        return gradient * sigmoidDerivative(self.state)

    def adjust(self, eta):
        pass

    def sigmoid(self, state):
        b = -6
        return 1.0 / (1 + np.exp(b * state))

    def sigmoidDerivative(self, state):
        return sigmoid(state) * (1 - sigmoid(state))
