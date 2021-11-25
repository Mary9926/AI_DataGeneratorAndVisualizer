import random
from neuron import *
from matplotlib import pyplot as plt
import tkinter
import numpy as np
from tkinter import *

root = tkinter.Tk()
root.title('Data')
root.geometry('500x500')
modesAmount = 1
samplesAmount = 0
meshgridPoints = 50
neuron = None
neuralNetwork = None
linear = None
activation = None
classes = []


class Point:
    X = 0
    Y = 0

    def __init__(self, x, y):
        self.X = x
        self.Y = y


class Class:
    Mode = Point(0, 0)
    xSamples = []  # array of x coordination
    ySamples = []  # array of y coordination
    Label = None

    def __init__(self, mode, xSamples, ySamples, label):
        self.Mode = mode
        self.xSamples = xSamples
        self.ySamples = ySamples
        self.Label = label

    def getSamples(self):
        samples = []
        for i in range(len(self.xSamples)):
            samples.append([self.xSamples[i], self.ySamples[i], self.Label])
        return samples


def initNeuralNetwork():
    allSamples = []
    for i in range(0, 4):
        allSamples = allSamples + classes[i].getSamples()

    np.random.shuffle(allSamples)

    global neuralNetwork
    neuralNetwork = NeuralNetwork(allSamples)


def trainNeuralNetwork():
    xx, yy = initBoundary(neuralNetwork.inputSamples)
    neuralNetwork.trainNeuralNetwork()
    input = np.c_[xx.reshape(-1, 1), yy.reshape(-1, 1)]
    predictedLabels = neuralNetwork.forwardPropagation(input)
    predictedLabels = predictedLabels[:, 0].reshape(xx.shape)
    drawBoundary(predictedLabels, xx, yy)
    plt.show()


def initNeuron(activationFunction, activationFunctionDerivative):
    global neuron
    neuron = Neuron(activationFunction, activationFunctionDerivative)


def train():
    samples_0 = classes[0].getSamples()
    samples_1 = classes[1].getSamples()
    allSamples = samples_0 + samples_1
    np.random.shuffle(allSamples)
    expectedOutput = np.asmatrix(allSamples)[:, 2].T
    inputSamples = np.delete(allSamples, 2, 1)
    inputSamples = np.c_[inputSamples, np.ones(len(inputSamples)) * -1]

    xx, yy = initBoundary(inputSamples)
    neuron.trainNeuron(inputSamples, expectedOutput)
    predictedLabels = neuron.decisionBoundary(xx, yy).reshape((meshgridPoints, meshgridPoints))

    drawBoundary(predictedLabels, xx, yy)
    plt.show()


def initBoundary(inputSamples):
    minX = inputSamples[:, 0].min() - 0.5
    maxX = inputSamples[:, 0].max() + 0.5
    minY = inputSamples[:, 1].min() - 0.5
    maxY = inputSamples[:, 1].max() + 0.5
    x = np.linspace(minX, maxX, meshgridPoints)
    y = np.linspace(minY, maxY, meshgridPoints)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def drawBoundary(predictedOutput, xx, yy):
    fig = plt.figure()
    fig.clear()
    ax = fig.add_subplot()
    ax.contourf(xx, yy, predictedOutput)
    drawSamples(ax)


def drawSamples(ax):
    for i in range(0, 2):
        class_i = classes[i]
        ax.scatter(class_i.Mode.X, class_i.Mode.Y, color='magenta', marker='s', label='Numbers')
        ax.scatter(class_i.xSamples, class_i.ySamples, color='magenta', marker='.')

    for i in range(2, 4):
        class_i = classes[i]
        ax.scatter(class_i.Mode.X, class_i.Mode.Y, color='cyan', marker='s')
        ax.scatter(class_i.xSamples, class_i.ySamples, color='cyan', marker='.')


def generateClasses():
    global classes, modesAmount
    classes = []
    samplesAmount = samplesAmountSlider.get()

    for i in range(0, 2):
        x, y = generateModes(modesAmount)
        xn, yn = generateSamples(x, y, samplesAmount)
        mode = Point(x, y)
        classes.append(Class(mode, xn, yn, 0))

    for i in range(0, 2):
        x, y = generateModes(modesAmount)
        xn, yn = generateSamples(x, y, samplesAmount)
        mode = Point(x, y)
        classes.append(Class(mode, xn, yn, 1))


def generateSamples(xClass, yClass, samplesAmount):
    devRange = random.uniform(0, 0.2)
    xClassNormal = np.random.normal(xClass, devRange, samplesAmount)
    yClassNormal = np.random.normal(yClass, devRange, samplesAmount)
    return xClassNormal, yClassNormal


def generateModes(modesAmount):
    rng = np.random.default_rng()
    rangeFirst = -1
    rangeLast = 1
    xClass = rng.uniform(rangeFirst, rangeLast, modesAmount)
    yClass = rng.uniform(rangeFirst, rangeLast, modesAmount)
    return xClass, yClass


def plot():
    global classes
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set(title='Data Visualizer - modes and samples', xlabel='x', ylabel='y')

    for i in range(0, 2):
        class_i = classes[i]
        ax.scatter(class_i.Mode.X, class_i.Mode.Y, color='magenta', marker='s', label='Numbers')
        ax.scatter(class_i.xSamples, class_i.ySamples, color='magenta', marker='.')

    for i in range(2, 4):
        class_i = classes[i]
        ax.scatter(class_i.Mode.X, class_i.Mode.Y, color='cyan', marker='s')
        ax.scatter(class_i.xSamples, class_i.ySamples, color='cyan', marker='.')

    plt.gca().legend(("1 class modes", "1 class sample", "2 class modes", "2 class sample"), loc="best")
    plt.show()


samplesAmountSliderlabel = Label(root, text="Please select number of samples").pack()
samplesAmountSlider = Scale(root, from_=0, to=100, orient=HORIZONTAL)
samplesAmountSlider.pack()

buttonPlot = tkinter.Button(root, text="Generate Samples", command=lambda: generateClasses())
buttonPlot.pack()

buttonPlot = tkinter.Button(root, text="Plot", command=lambda: plot())
buttonPlot.pack()

buttonPlot = tkinter.Button(root, text="Init NeuralNetwork",
                            command=lambda: initNeuralNetwork())
buttonPlot.pack()

buttonPlot = tkinter.Button(root, text="Train NeuralNetwork", command=lambda: trainNeuralNetwork())
buttonPlot.pack()

# buttonPlot = tkinter.Button(root, text="Init Neuron with sigmoid activation function",
#                             command=lambda: initNeuron(sigmoid, sigmoidDerivative))
# buttonPlot.pack()
#
# buttonPlot = tkinter.Button(root, text="Init Neuron with heaviside activation function",
#                             command=lambda: initNeuron(heaviside, heavisideDerivative))
# buttonPlot.pack()
#
# buttonPlot = tkinter.Button(root, text="Init Neuron with sin activation function",
#                             command=lambda: initNeuron(sin, sinDerivative))
# buttonPlot.pack()
#
# buttonPlot = tkinter.Button(root, text="Init Neuron with tanh activation function",
#                             command=lambda: initNeuron(tanh, tanhDerivative))
# buttonPlot.pack()
#
# buttonPlot = tkinter.Button(root, text="Train Neuron", command=lambda: train())
# buttonPlot.pack()

mainMenu = tkinter.Menu()
root.config(menu=mainMenu)
dataMenu = tkinter.Menu(mainMenu)
root.mainloop()
