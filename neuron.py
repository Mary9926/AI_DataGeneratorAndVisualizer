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
        print("Input Samples: ")
        print(inputSamples)
        print("Init Weights: ")
        print(self.weights)
        print("Expected Output: ")
        #expectedOutput = self.weights[:, np.newaxis].T @ inputSamples.T
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
            print(self.activationFunctionDerivative(self.neuronState(inputSamples)).reshape(len(inputSamples), 1).shape)
            print(self.activationFunctionDerivative(self.neuronState(inputSamples)).reshape(len(inputSamples), 1))
            error = np.mean(error)
            adjustments = error * self.activationFunctionDerivative(self.neuronState(inputSamples)).reshape(len(inputSamples), 1) * inputSamples
            print("Adjustments: ")
            print(adjustments)
            adjustments = np.mean(adjustments, 0)
            #adjustments = adjustments.T
            self.weights += adjustments
            if np.all(adjustments <= self.epsilon):
                break

    def learnNeuron(self, inputSamples):
        return self.activationFunction(self.neuronState(inputSamples))

    def generateBoundary(self, prediction, xx, yy):  # rysowanie
        self.fig.clear()
        ax = self.fig.add_subplot()
        ax.contourf(xx, yy, prediction)  # sam określa kolorki
        self.drawSamples("red", "darkRed", "green", "darkGreen", ax)

    def prepareDecisionBoundary(self, inputSamples):
        meshgridPoints = 50
        minX = inputSamples[:, 0], np.array([0])
        print('minX')
        print(minX)
        maxX = np.max(np.append(inputSamples[:, 0], np.array([1])))
        print('maxX')
        print(maxX)
        minY = np.min(np.append(inputSamples[:, 1], np.array([0])))
        print('minY')
        print(minY)
        maxY = np.max(np.append(inputSamples[:, 1], np.array([1])))
        print('maxY')
        print(maxY)
        x = np.linspace(minX, maxX, meshgridPoints)
        y = np.linspace(minY, maxY, meshgridPoints)
        xx, yy = np.meshgrid(x, y)
        print('x')
        print(x)
        print('y')
        print(y)
        print('xx')
        print(xx)
        print('yy')
        print(yy)

        return xx, yy


def sigmoid(s):
    b = -6
    return 1.0 / (1 + np.exp(b * s))


def sigmoidDerivative(s):
    return sigmoid(s) * (1 - sigmoid(s))


def heaviside(s):
    return 0 if s < 0 else 1


def heavisideDerivative(s):
    return 1


