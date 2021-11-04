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


def initNeuron(activationFunction, activationFunctionDerivative):
    global neuron
    neuron = Neuron(activationFunction, activationFunctionDerivative)


def train():
    xd


def generate():
    global classes, modesAmount
    classes = []
    samplesAmount = samplesAmountSlider.get()
    x, y = generateModes(modesAmount)
    xn, yn = generateSamples(x, y, samplesAmount)
    classes.append([x, y, xn, yn, "0"])
    x, y = generateModes(modesAmount)
    xn, yn = generateSamples(x, y, samplesAmount)
    classes.append([x, y, xn, yn, "1"])
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
    #classLabel = random.choice([0, 1])
    return xClass, yClass


# def plot():
#     samplesAmount = samplesAmountSlider.get()
#     fig = plt.figure()
#     ax = fig.add_subplot()
#     ax.set(title='Data Visualizer - modes and samples', xlabel='x', ylabel='y')
#
#     x, y = generateModes(modesAmount)
#     xn, yn = generateSamples(x, y, samplesAmount)
#     for i in range(modesAmount):
#         ax.scatter(x, y, color='magenta', marker='s', label='Numbers')
#         ax.scatter(xn, yn, color='magenta', marker='.')
#
#     x, y = generateModes(modesAmount)
#     xn, yn = generateSamples(x, y, samplesAmount)
#     for i in range(modesAmount):
#         ax.scatter(x, y, color='cyan', marker='s')
#         ax.scatter(xn, yn, color='cyan', marker='.')
#
#     plt.gca().legend(("1 class modes", "1 class sample", "2 class modes", "2 class sample"), loc="best")
#     plt.show()


def plot():
    global classes
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set(title='Data Visualizer - modes and samples', xlabel='x', ylabel='y')

    class_0 = classes[0]
    ax.scatter(class_0[0], class_0[1], color='magenta', marker='s', label='Numbers')
    ax.scatter(class_0[2], class_0[3], color='magenta', marker='.')

    class_1 = classes[1]
    ax.scatter(class_1[0], class_1[1], color='cyan', marker='s')
    ax.scatter(class_1[2], class_1[3], color='cyan', marker='.')

    plt.gca().legend(("1 class modes", "1 class sample", "2 class modes", "2 class sample"), loc="best")
    plt.show()


samplesAmountSliderlabel = Label(root, text="Please select number of samples").pack()
samplesAmountSlider = Scale(root, from_=0, to=100, orient=HORIZONTAL)
samplesAmountSlider.pack()

buttonPlot = tkinter.Button(root, text="Generate Samples", command=lambda: generate())
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
