from window import *
from neuron import *

if __name__ == '__main__':

    neuron = Neuron()

    print("Beginning Randomly Generated Weights: ")
    print(neuron.weights)

    # training data consisting of 4 examples--3 input values and 1 output
    trainingInputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])

    trainingOutputs = np.array([[0, 1, 1, 0]]).T

    # training taking place
    neuron.trainNeuron(trainingInputs, trainingOutputs, 15000)





