# from perceptron import Perceptron
import numpy as np


training_data_xor = [[np.array([[0], [0]]), np.array([0])],
                     [np.array([[0], [1]]), np.array([1])],
                     [np.array([[1], [0]]), np.array([1])],
                     [np.array([[1], [1]]), np.array([0])]]

training_data_and = [[np.array([[0], [0]]), np.array([0])],
                     [np.array([[0], [1]]), np.array([0])],
                     [np.array([[1], [0]]), np.array([0])],
                     [np.array([[1], [1]]), np.array([1])]]

training_data_3and = [[np.array([[0], [0], [0]]), np.array([0])],
                      [np.array([[1], [0], [0]]), np.array([0])],
                      [np.array([[0], [1], [0]]), np.array([0])],
                      [np.array([[0], [0], [1]]), np.array([0])],
                      [np.array([[1], [1], [0]]), np.array([0])],
                      [np.array([[0], [1], [1]]), np.array([0])],
                      [np.array([[1], [0], [1]]), np.array([0])],
                      [np.array([[1], [1], [1]]), np.array([1])]]

training_data_3or = [[np.array([[0], [0], [0]]), np.array([0])],
                     [np.array([[1], [0], [0]]), np.array([1])],
                     [np.array([[0], [1], [0]]), np.array([1])],
                     [np.array([[0], [0], [1]]), np.array([1])],
                     [np.array([[1], [1], [0]]), np.array([1])],
                     [np.array([[0], [1], [1]]), np.array([1])],
                     [np.array([[1], [0], [1]]), np.array([1])],
                     [np.array([[1], [1], [1]]), np.array([1])]]


training_data_not = [[np.array([[0]]), np.array([1])],
                     [np.array([[1]]), np.array([0])]]


def relu(x):
    return x * (x > 0)

def drelu(x):
    return 1. * (x > 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# to jest pochodna funkcji
def dsigmoid(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, numI, numH, numO):
        self.inputs_nodes = numI
        self.hidden_nodes = numH
        self.output_nodes = numO
        self.learning_rate = 0.05

        self.weights_ih = np.random.rand(self.hidden_nodes, self.inputs_nodes) * 2 - 1
        self.weights_ho = np.random.rand(self.output_nodes, self.hidden_nodes) * 2 - 1

        self.bias_h = np.random.rand(self.hidden_nodes, 1) * 2 - 1
        self.bias_o = np.random.rand(self.output_nodes, 1) * 2 - 1


    def predict(self, inp):
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

        # deltas
        weight_ho_deltas = gradients * np.transpose(hidden)

        # Adjust the weights by deltas
        self.weights_ho += weight_ho_deltas
        # Adjust the bias by its deltas(which is just the gradients)
        self.bias_o += gradients

        # calculate hidden layer errors
        hidden_errors = self.weights_ho.T.dot(output_errors)

        # calculate hidden gradient
        hidden_gradient = dsigmoid(hidden) * hidden_errors * self.learning_rate

        # calculate input -> hidden deltas
        weight_ih_deltas = hidden_gradient * np.transpose(inp)

        # // Adjust the weights by deltas
        self.weights_ih += weight_ih_deltas

        # Adjust the bias by its deltas(which is just the gradients)
        self.bias_h += hidden_gradient

        pass


def ucz(per, tr_data):
    for k in range(10000):
        for tr in tr_data:
            per.train(tr[0], tr[1])
            print " tutaj powinno byc  " + str(tr[1]) + " a jest " + str(per.predict(tr[0]))


# nn = NeuralNetwork(2, 2, 1)
# # musi wejsc wektor pionowy
# # inpu = np.array([[1], [1]])
# # print nn.predict(inpu)
#
# ucz(nn, training_data_xor)



