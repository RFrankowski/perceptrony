import numpy as np


def ucz(per, tr_data):
    for k in range(50):
        for tr in tr_data:
            i = tr
            print i[0]
            print i[1]
            per.train(np.array(i[0]), np.array(i[1]))
            print " tutaj powinno byc  " + str(i[1]) + " a jest " + str(per.query(i[0])) +"\n"

def matrixsigmoida(x):
    return 1 / (1 + np.exp(-x))


class Neuron(object):
    def __init__(self, input, hiddeinput, output, lrrate):
        self.ino = input
        self.hid = hiddeinput
        self.out = output

        self.wih = np.random.normal(0.0, pow(self.ino, -0.5), (self.hid, self.ino))
        self.who = np.random.normal(0.0, pow(self.hid, -0.5), (self.out, self.hid))

        self.lr = lrrate

        self.activiation_function = matrixsigmoida

        pass

    def train(self, input_list, expected):
        input = np.array(input_list).T
        target = np.array(expected).T


        hidden_input = np.dot(self.wih, input)
        hidden_output = self.activiation_function(hidden_input)

        final_input = np.dot(self.who, hidden_output)
        final_output = self.activiation_function(final_input)

        output_error = target - final_output
        hidden_error = np.dot(self.who.T, output_error)

        self.who += self.lr * np.dot((output_error * final_output * (1.0  - final_output)), np.transpose(hidden_output))

        self.wih += self.lr * np.dot((hidden_error * hidden_output * (1.0 - hidden_output)), np.transpose(input))

        pass

    def query(self, input_list):
        input = np.array(input_list, ndmin=2).T

        hidden_input = np.dot(self.wih, input)
        hidden_output = self.activiation_function(hidden_input)

        final_input = np.dot(self.who, hidden_output)
        final_output = self.activiation_function(final_input)

        return final_output

n = Neuron(2, 2, 1, 0.1)
# training_data_xor = [[[0, 0, 1], 0], [[0, 1, 1], 1], [[1, 0, 1], 1], [[1, 1, 1], 0]]
training_data_xor = [[np.array([[0], [0]]), np.array([0])], [np.array([[0], np.array([1])]), np.array([1])],
                     [np.array([[1], [0]]), np.array([1])],
                     [np.array([[1], [1]]), np.array([0])]]
# n.train(np.array([[1],[0]]),np.array([1]))

print n.query(np.array([1,0]))
ucz(n,training_data_xor)
