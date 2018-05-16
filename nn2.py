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
                     [np.array([[0]]), np.array([0])]]


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
    def __init__(self, numI, numFH, numH, numO):
        self.inputs_nodes = numI
        self.first_layer_hidden_nodes = numFH
        self.second_layer_hidden_nodes = numH
        self.output_nodes = numO
        self.learning_rate = 0.05

        self.weights_fh = np.random.rand(self.first_layer_hidden_nodes, self.inputs_nodes) * 2 - 1
        self.weights_sh = np.random.rand(self.second_layer_hidden_nodes, self.first_layer_hidden_nodes) * 2 - 1
        self.weights_ho = np.random.rand(self.output_nodes, self.second_layer_hidden_nodes) * 2 - 1

        self.bias_fh = np.random.rand(self.first_layer_hidden_nodes, 1) * 2 - 1
        self.bias_sh = np.random.rand(self.second_layer_hidden_nodes, 1) * 2 - 1
        self.bias_o = np.random.rand(self.output_nodes, 1) * 2 - 1

    def predict(self, inp):
        first_hidden = self.weights_fh.dot(inp) + self.bias_fh
        first_hidden = sigmoid(first_hidden)

        second_hidden = self.weights_sh.dot(first_hidden) + self.bias_sh
        second_hidden = sigmoid(second_hidden)

        output = self.weights_ho.dot(second_hidden) + self.bias_o
        output = sigmoid(output)

        return output

    def train(self, inp, targets):
        first_hidden = self.weights_fh.dot(inp) + self.bias_fh
        first_hidden = sigmoid(first_hidden)

        second_hidden = self.weights_sh.dot(first_hidden) + self.bias_sh
        second_hidden = sigmoid(second_hidden)

        output = self.weights_ho.dot(second_hidden) + self.bias_o
        outputs = sigmoid(output)

        # output layer
        output_errors = targets - outputs
        gradients = dsigmoid(outputs) * output_errors * self.learning_rate
        weight_ho_deltas = gradients * np.transpose(second_hidden)
        self.weights_ho += weight_ho_deltas
        self.bias_o += gradients

        # calculate second_hidden layer errors
        hidden_errors = self.weights_ho.T.dot(output_errors)
        hidden_gradient = dsigmoid(second_hidden) * hidden_errors * self.learning_rate
        weight_ih_deltas = hidden_gradient * np.transpose(first_hidden)
        self.weights_sh += weight_ih_deltas
        self.bias_sh += hidden_gradient


        hidden_errors = self.weights_sh.T.dot(hidden_errors)
        hidden_gradient = dsigmoid(first_hidden) * hidden_errors * self.learning_rate
        weight_ih_deltas = hidden_gradient * np.transpose(inp)
        self.weights_fh += weight_ih_deltas
        self.bias_fh += hidden_gradient

def ucz(per, tr_data):
    for k in range(10000):
        for tr in tr_data:
            per.train(tr[0], tr[1])
            print " tutaj powinno byc  " + str(tr[1]) + " a jest " + str(per.predict(tr[0]))


nn = NeuralNetwork(3, 3, 2, 1)
ucz(nn, training_data_3or)
