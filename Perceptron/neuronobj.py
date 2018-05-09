import numpy as np
import random


def matrixsigmoida(X):
    sigmoida = lambda x: 1.0 / (1.0 + np.exp(-x))
    (a, b) = np.shape(X)
    for i in range(0, a):
        for j in range(0, b):
            X[i][j] = sigmoida(X[i][j])

    return (X)


class neuron(object):
    def __init__(self, input, hiddeinput, output, lrrate):
        self.ino = input
        self.hid = hiddeinput
        self.out = output
        self.wih = []
        self.who = []
        for i in range(input):
            self.wih.append(random.random())
            self.who.append(random.random())

        # print self.who
        # print self.wih

        self.lr = lrrate

        self.activiation_function = matrixsigmoida

        pass

    def train(self, input_list, expected):
        input = np.array(input_list, ndmin=2).T
        target = np.array(expected, ndmin=2).T

        hidden_input = np.dot(self.wih, input)
        hidden_output = self.activiation_function(hidden_input)

        final_input = np.dot(self.who, hidden_output)
        final_output = self.activiation_function(final_input)

        output_error = target - final_output
        hidden_error = np.dot(self.who.T, output_error)

        self.who += self.lr * np.dot((output_error * final_output * (1.0 - final_output)), np.transpose(hidden_output))

        self.wih += self.lr * np.dot((hidden_error * hidden_output * (1.0 - hidden_output)), np.transpose(input))
        pass

    def query(self, input_list):
        input = np.array(input_list, ndmin=2).T

        hidden_input = np.dot(self.wih, input)
        hidden_output = self.activiation_function(hidden_input)

        final_input = np.dot(self.who, hidden_output)
        final_output = self.activiation_function(final_input)

        return final_output


n = neuron(2, [[1, 1], [1, 1]], 1, 0.001)

n.train([[1, 2], [1, 1]], [2, 2])
