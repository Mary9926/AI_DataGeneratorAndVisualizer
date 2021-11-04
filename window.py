import random
from neuron import  *
from matplotlib import pyplot as plt
import tkinter
import numpy as np
from tkinter import *

root = tkinter.Tk()
root.title('Data')
root.geometry('500x500')
modesAmount = 1
samplesAmount = 0

def initNeuron():
    Neuron()





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
    classLabel = random.choice([0, 1])
    return xClass, yClass, classLabel


def plot(): 
    samplesAmount = samplesAmountSlider.get()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set(title='Data Visualizer - modes and samples', xlabel='x', ylabel='y')

    x, y, label = generateModes(modesAmount)
    xn, yn = generateSamples(x, y, samplesAmount)
    for i in range(modesAmount):
        ax.scatter(x, y, color='magenta', marker='s', label='Numbers')
        ax.scatter(xn, yn, color='magenta', marker='.')

    x, y, label = generateModes(modesAmount)
    xn, yn = generateSamples(x, y, samplesAmount)
    for i in range(modesAmount):
        ax.scatter(x, y, color='cyan', marker='s')
        ax.scatter(xn, yn, color='cyan', marker='.')

    plt.gca().legend(("1 class modes", "1 class sample", "2 class modes", "2 class sample"), loc="best")
    plt.show()


samplesAmountSliderlabel = Label(root, text="Please select number of samples").pack()
samplesAmountSlider = Scale(root, from_=0 ,to=100, orient= HORIZONTAL)
samplesAmountSlider.pack()

buttonPlot = tkinter.Button(root, text="Plot", command=lambda: plot())
buttonPlot.pack()

buttonPlot = tkinter.Button(root, text="Init Neuron", command=lambda: initNeuron())
buttonPlot.pack()

buttonPlot = tkinter.Button(root, text="Train neuron", command=lambda: plot())
buttonPlot.pack()


mainMenu = tkinter.Menu()
root.config(menu=mainMenu)
dataMenu = tkinter.Menu(mainMenu)
root.mainloop()

