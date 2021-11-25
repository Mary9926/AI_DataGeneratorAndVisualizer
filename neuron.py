import numpy as np
from window import *


def addEmptyColumn(inputSamples):
    return np.c_[inputSamples, np.ones(len(inputSamples)) * -1]


def deleteLastColumn(x):
    return np.delete(x, len(x[0]) - 1, 1)


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

    def __init__(self, samples):
        neuronAmountInInputLayer = 2
        neuronAmountInHiddenLayer = 3
        neuronAmountInOutputLayer = 2
        labelAmount = 1
        self.layers = [Linear(neuronAmountInInputLayer, 2 + labelAmount), Activation(),
                       Linear(neuronAmountInHiddenLayer, neuronAmountInInputLayer + labelAmount), Activation(),
                       Linear(neuronAmountInOutputLayer, neuronAmountInHiddenLayer + labelAmount), Activation()]
        self.expectedOutput = self.getExpectedOutput(samples)

        inputSamples = np.delete(samples, 2, 1)
        #inputSamples = addEmptyColumn(inputSamples)
        self.inputSamples = inputSamples

    def getExpectedOutput(self, inputSamples):
        inputSamples = np.asarray(inputSamples)
        expectedOutput = inputSamples[:, 2].T
        return np.expand_dims(expectedOutput, -1)

    def forwardPropagation(self, inputSamples):
        predictedOutput = inputSamples
        for layer in self.layers:
            predictedOutput = layer.forwardPropagation(predictedOutput)
        return predictedOutput

    def trainNeuralNetwork(self):
        for epoch in range(self.epochs):
            for i in range(0, len(self.inputSamples)):
                predictedOutput = self.forwardPropagation(self.inputSamples)
                sub = np.subtract(self.expectedOutput, predictedOutput)
                error = np.square(sub).mean()
                gradient = error * predictedOutput
                for layer in self.layers[::-1]: #reverse order
                    gradient = layer.backPropagation(gradient)
                    layer.adjust(self.eta)


class Linear:
    
    def __init__(self, neuronAmount, inputWeightsAmount):
        rng = np.random.default_rng()
        self.weights = []
        self.neuronAmount = neuronAmount
        self.inputSamples = []
        self.gradient = []

        for i in range(0, neuronAmount):
            w = rng.random(inputWeightsAmount)
            self.weights.append(w)
        self.weights = np.array(self.weights)

    def forwardPropagation(self, inputSamples):
        inputSamples = addEmptyColumn(inputSamples)
        self.inputSamples = inputSamples
        return self.inputSamples @ self.weights.T

    def backPropagation(self, gradient):
        self.gradient = gradient
        weights = deleteLastColumn(self.weights)
        return self.gradient @ weights

    def adjust(self, eta):
        #self.inputSamples = np.delete(self.inputSamples, 3, 1)
        self.weights += eta * (self.gradient.T @ self.inputSamples)


class Activation:

    def __init__(self):
        self.state = []

    def forwardPropagation(self, state):
        self.state = state
        return self.sigmoid(state)

    def backPropagation(self, gradient):
        return self.sigmoidDerivative(self.state) * gradient

    def adjust(self, eta):
        pass

    def sigmoid(self, state):
        b = -6
        return 1.0 / (1 + np.exp(b * state))

    def sigmoidDerivative(self, state):
        onesColumn = np.ones(state.shape)
        return self.sigmoid(state) * (onesColumn - self.sigmoid(state))
