import random
from matplotlib import pyplot as plt
import tkinter
import numpy as np

root = tkinter.Tk()
root.title('Data')
root.geometry('500x500')


def generateSamples(xClass, yClass, samplesAmount, devRange):
    xClassNormal = np.random.normal(xClass, devRange, samplesAmount)
    yClassNormal = np.random.normal(yClass, devRange, samplesAmount)
    return xClassNormal, yClassNormal


def generateModes(modesAmount):
    rng = np.random.default_rng()
    rangeFirst = 0
    rangeLast = 1
    xClass = rng.uniform(rangeFirst, rangeLast, modesAmount)
    yClass = rng.uniform(rangeFirst, rangeLast, modesAmount)
    classLabel = random.choice([0, 1])
    return xClass, yClass, classLabel


def plot():
    fig = plt.figure()
    axis = fig.add_subplot()
    modesAmount = 1
    samplesAmount = 4
    devRange = 0.25
    x, y, label = generateModes(modesAmount)

    xn, yn = generateSamples(x, y, samplesAmount, devRange)

    for i in range(modesAmount):
        axis.scatter(x, y, color='magenta', marker='*')
        axis.scatter(xn, yn, color='magenta', marker='.')

    x, y, label = generateModes(modesAmount)

    xn, yn = generateSamples(x, y, samplesAmount, devRange)
    for i in range(modesAmount):
        axis.scatter(x, y, color='cyan', marker='*')
        axis.scatter(xn, yn, color='cyan', marker='.')

    plt.show()


buttonPlot = tkinter.Button(root, text="Plot", command=lambda: plot())
buttonPlot.pack()

mainMenu = tkinter.Menu()
root.config(menu=mainMenu)
dataMenu = tkinter.Menu(mainMenu)
root.mainloop()
