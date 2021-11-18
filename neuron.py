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
    layers = 3

    def __init__(self, layers):
        rng = np.random.default_rng()
        weights = []
        self.layers = layers

        for i in range(len(layers) - 1 ):
            w = rng.random(layers[i] + 1, layers[i] - 1) # weights in range (-1, 1)
            weights.append(w)
            self.weights = weights

    def backPropagation(self): # for training
        # for i in np.arange(0,len(leyers) - 2):
        #     w = np.random.randn(layers[i] + 1, layers[i + 1] + 1 )
        #     self.W.append(w/ np.sqrt(layers[i]))



    def forwardPropagation(self, inputSamples): # for evaluation give us a prediction output predicted by network
        for i in enumerate(self.weights):
            networkInputs = inputSamples @ weigths
            activations = self.activationFunction(networkInputs)
            print('activations')
            print(activations)
            return activations

    def gradientOfActivationFunction(self, learningRate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            activationFunctionDerivative = self.activationFunctionDerivative[i]
            weights+= activationFunctionDerivative + learningRate



class Linear:
    neuronNumber = 2
    def __init__(self):
        self.inputSamples = []
        self.neuronNumber = neuronNumber

    def forwardPropagation(self, inputSamples):
        self.inputSamples = np.c_[inputSamples, np.ones(len(inputSamples)) * -1]
        return  inputSamples @ weigths #state

    def backPropagation(self, gradientOfLoss):
        self.gradientOfLoss = gradientOfLoss
        return gradientOfLoss @ weights

    def adjust(self, eta): # here ask
        self.weights += eta * gradientOfLoss * inputSamples


class Activation:

    def __init__(self, state):
        self.state = state

    def forwardPropagation(self, state):
       self.state = state
       return sigmoid(state)

    def backPropagation(self, gradientOfLoss):
        return gradientOfLoss * sigmoidDerivative(state)

    def adjust(self, eta):
        pass


def sigmoid(s):
    b = -6
    return 1.0 / (1 + np.exp(b * s))


def sigmoidDerivative(s):
    return sigmoid(s) * (1 - sigmoid(s))


