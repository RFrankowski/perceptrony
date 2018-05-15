# from perceptron import Perceptron
import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, numI, numH, numO):
        self.inputs_nodes = numI
        self.hidden_nodes = numH
        self.output_nodes = numO
        self.learning_rate = 0.1

        # self.weights_ih = np.zeros((self.hidden_nodes, self.inputs_nodes))
        # self.weights_ho = np.zeros((self.output_nodes, self.hidden_nodes))

        self.weights_ih = np.random.rand(self.hidden_nodes, self.inputs_nodes) * 2 - 1
        self.weights_ho = np.random.rand(self.output_nodes, self.hidden_nodes) * 2 - 1

        # self.bias_h = np.zeros((self.hidden_nodes, 1))
        # self.bias_o = np.zeros((self.output_nodes, 1))

        self.bias_h = np.random.rand(self.hidden_nodes, 1) * 2 - 1
        self.bias_o = np.random.rand(self.output_nodes, 1) * 2 - 1

        # print self.weights_ih

    def feedForward(self, inp):
        hidden = self.weights_ih.dot(inp) + self.bias_h
        # dodanie funkcji aktywacji
        hidden = sigmoid(hidden)

        # warstwa wyjsciowa
        output = self.weights_ho.dot(hidden) + self.bias_o
        output = sigmoid(output)

        return output

    def train(self, inp, targets):
        hidden = self.weights_ih.dot(inp) + self.bias_h
        # dodanie funkcji aktywacji
        hidden = sigmoid(hidden)

        # warstwa wyjsciowa
        output = self.weights_ho.dot(hidden) + self.bias_o
        outputs = sigmoid(output)

        # Calculate error
        output_errors = targets - outputs

        # gradient

        gradients = dsigmoid(outputs) * output_errors * self.learning_rate
        # print gradients
        # deltas

        weight_ho_deltas = gradients * np.transpose(hidden)
        # print  weight_ho_deltas

        # // Adjust the weights by deltas
        self.weights_ho = self.weights_ho + weight_ho_deltas
        # Adjust the bias by its deltas(which is just the gradients)
        self.bias_o = self.bias_o + gradients

        # calculate hidden layer errors
        hidden_errors = self.weights_ho.T.dot(output_errors)

        # calculate hidden gradient
        hidden_gradient = dsigmoid(hidden) * hidden_errors * self.learning_rate

        # calculate input -> hidden delatas
        weight_ih_deltas = hidden_gradient * np.transpose(inp)

        self.weights_ih += weight_ih_deltas

        self.bias_h += hidden_gradient

        pass


training_data_xor = [[np.array([[0], [0]]), np.array([0])],
                     [np.array([[0], [1]]), np.array([1])],
                     [np.array([[1], [0]]), np.array([1])],
                     [np.array([[1], [1]]), np.array([0])]]


def ucz(per, tr_data):
    for k in range(100000):
        for tr in tr_data:
            i = tr
            per.train(i[0], i[1])
            print " tutaj powinno byc  " + str(i[1]) + " a jest " + str(per.feedForward(i[0]))


nn = NeuralNetwork(2, 2, 1)
# musi wejsc wektor pionowy
# inpu = np.array([[1], [1]])



# print nn.feedForward(inpu)

ucz(nn, training_data_xor)



# print inpu
# out = nn.feedForward(inpu)
# print np.zeros((2,4))
