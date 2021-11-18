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
    epochs = 10000
    eta = 0.1


    def __init__(self, layers, neuron1, neuron2):
        rng = np.random.default_rng()
        weights = []
        #self.layers = [inputLayer, hiddenLayer, outputLayer]
        self.layers = [Linear, Activation, Linear, Activation, Linear, Activation]
        self.neuron1 = neuron1
        self.neuron2 = neuron2
        neuronAmount = 2
       # for i in range(len(layers) - 1):

        for i in range(0, neuronAmount):
          #  w = rng.random(layers[i] + 1, layers[i] - 1)  # weights in range (-1, 1)
            w = rng.random(3)
            weights.append(w)
            self.weights = weights

    def forwardPropagation(self, inputSamples):
        for layer in self.layers:
            inputSamples = layer.forward(inputSamples)
        return inputSamples

    def backPropagation(self, inputSamples, expectedLabels):
        for i in range(self.epochs):
            predictedLabels = self.forwardPropagation(inputSamples)
            gradient = np.mean((expectedLabels - predictedLabels) * (expectedLabels - predictedLabels))
            for layer in self.layers[::-1]: #reverse order
                gradient = layer.back(gradient)
                predictedLabels = layer.forward(predictedLabels)
                layer.adjust(eta)

    def adjustments(self):
        for layer in self.layers:
            layer.adjust(eta)


class Linear:
    def __init__(self):
        self.inputSamples = []

    def forwardPropagation(self, inputSamples):
        return inputSamples @ weigths #state

    def backPropagation(self, gradient):
        self.gradient = gradient
        return gradient @ weights

    def adjust(self, eta): # here ask
        self.weights += eta * gradientOfLoss * inputSamples


class Activation:

    def __init__(self, state):
        self.state = state

    def forwardPropagation(self, state):
       self.state = state
       return sigmoid(state)

    def backPropagation(self, gradient):
        return gradient * sigmoidDerivative(state)

    def adjust(self, eta):
        pass



def sigmoid(s):
    b = -6
    return 1.0 / (1 + np.exp(b * s))


def sigmoidDerivative(s):
    return sigmoid(s) * (1 - sigmoid(s))


    #
    # def forwardPropagation(self, inputSamples): # for evaluation give us a prediction output predicted by network
    #     for i in enumerate(self.weights):
    #         networkInputs = inputSamples @ weigths
    #         activations = self.activationFunction(networkInputs)
    #         print('activations')
    #         print(activations)
    #         return activations
    #
    # def backPropagation(self):  # for training
    #
    # # for i in np.arange(0,len(leyers) - 2):
    # #     w = np.random.randn(layers[i] + 1, layers[i + 1] + 1 )
    # #     self.W.append(w/ np.sqrt(layers[i]))
    #
    # def gradientOfActivationFunction(self, learningRate):
    #     for i in range(len(self.weights)):
    #         weights = self.weights[i]
    #         activationFunctionDerivative = self.activationFunctionDerivative[i]
    #         weights+= activationFunctionDerivative + learningRate
