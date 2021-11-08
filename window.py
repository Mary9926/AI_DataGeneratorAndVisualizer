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
neuron = None
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
    #Label = ""
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
    #neuron.prepareDecisionBoundary(inputSamples)
    neuron.trainNeuron(inputSamples, expectedOutput)


def generateClasses():
    global classes, modesAmount
    classes = []
    samplesAmount = samplesAmountSlider.get()

    x, y = generateModes(modesAmount)
    xn, yn = generateSamples(x, y, samplesAmount)
    mode = Point(x, y)
    #classes.append(Class(mode, xn, yn, "0"))
    classes.append(Class(mode, xn, yn, 0))

    x, y = generateModes(modesAmount)
    xn, yn = generateSamples(x, y, samplesAmount)
    mode = Point(x, y)
    #classes.append(Class(mode, xn, yn, "1"))
    classes.append(Class(mode, xn, yn, 1))
    print(classes)


def generateSamples(xClass, yClass, samplesAmount):
    devRange = random.uniform(0, 1)
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

    class_0 = classes[0]
    ax.scatter(class_0.Mode.X, class_0.Mode.Y, color='magenta', marker='s', label='Numbers')
    ax.scatter(class_0.xSamples, class_0.ySamples, color='magenta', marker='.')

    class_1 = classes[1]
    ax.scatter(class_1.Mode.X, class_1.Mode.Y, color='cyan', marker='s')
    ax.scatter(class_1.xSamples, class_1.ySamples, color='cyan', marker='.')

    plt.gca().legend(("1 class modes", "1 class sample", "2 class modes", "2 class sample"), loc="best")
    plt.show()


samplesAmountSliderlabel = Label(root, text="Please select number of samples").pack()
samplesAmountSlider = Scale(root, from_=0, to=100, orient=HORIZONTAL)
samplesAmountSlider.pack()

buttonPlot = tkinter.Button(root, text="Generate", command=lambda: generateClasses())
buttonPlot.pack()

buttonPlot = tkinter.Button(root, text="Plot", command=lambda: plot())
buttonPlot.pack()

buttonPlot = tkinter.Button(root, text="Init Neuron", command=lambda: initNeuron(sigmoid, sigmoidDerivative))
buttonPlot.pack()

buttonPlot = tkinter.Button(root, text="Train Neuron", command=lambda: train())
buttonPlot.pack()

mainMenu = tkinter.Menu()
root.config(menu=mainMenu)
dataMenu = tkinter.Menu(mainMenu)
root.mainloop()
