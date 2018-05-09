import random
import numpy as np
import math


# to jest funkcja aktywacji
def sign(n):
    if n >= 0:
        return 1
    else:
        return -1


def matrixsigmoida(x):
    return 1 / (1 + math.exp(-x))

    # return (sigmoida(X))


# 0.752135511929
# [0.469757865067298, 0.1820034389739238, -0.5845975069602078]

class Perceptron:
    def __init__(self):
        self.weigths = []
        self.lr = random.random()
        for i in range(3):
            self.weigths.append(random.random())

    def guess(self, inputs):
        summ = 0
        for w in range(len(self.weigths)):
            summ += inputs[w] * self.weigths[w]
            # print summ
        output = sign(summ)
        return output

    def train(self, inputs, target):
        guess = self.guess(inputs)
        error = target - guess
        # print guess
        # print error
        # korygowanie wag
        for i in range(len(self.weigths)):
            self.weigths[i] += error * inputs[i] * self.lr
            # print self.weigths


# print p.guess([-5, 45])
# and training data
training_data = [[[0, 0, 1], 0], [[0, 1, 1], 0], [[1, 0, 1], 0], [[1, 1, 1], 1]]

training_data_or = [[[0, 0, 1], 0], [[0, 1, 1], 1], [[1, 0, 1], 1], [[1, 1, 1], 1]]

training_data_not = [[[0], 1], [[1], 0]]


# print p.guess([0, 0, 1])


def ucz(per, tr_data):
    # print tr_data
    # print tr_data[2][1]
    for k in range(100):
        i = random.choice(tr_data)
        per.train(i[0], i[1])

    for li in tr_data:
        print " tutaj powinno byc  " + str(li[1]) + " a jest " + str(per.guess(li[0]))


# perceptron dla and
# p = Perceptron()
# ucz(p, training_data)
# print p.lr
# print p.weigths

# gotowy perceptron do pokazania dla and
# pAnd = Perceptron()
#
# pAnd.weigths = [0.469757865067298, 0.1820034389739238, -0.5845975069602078]
# pAnd.lr = 0.752135511929
# print pAnd.guess([1, 1, 1])


# ptrainOr = Perceptron()
# ucz(ptrainOr, training_data_or)
# print ptrainOr.lr
# print ptrainOr.weigths
#
# ptrainOr.lr = 0.167631213692
# ptrainOr.weigths = [0.6586391948364184, 0.43477785666346414, -0.10141305826027047]


