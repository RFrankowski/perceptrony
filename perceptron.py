import random

import math


# to jest funkcja aktywacji
def sign(n):
    if n >= 0:
        return 1
    else:
        return -1


def unipolar(n):
    if n >= 0:
        return 1
    else:
        return 0


def matrixsigmoida(x):
    return 1 / (1 + math.exp(-x))

    # return (sigmoida(X))


class Perceptron:
    def __init__(self):
        self.weigths = []
        self.lr = 0.01

    def setRandomWeigths(self, inp):
        if not self.weigths:
            for i in range(len(inp)):
                self.weigths.append((random.random() * (-2)) - 1)

    def guess(self, inputs):
        self.setRandomWeigths(inputs)
        summ = 0
        for w in range(len(self.weigths)):
            summ += inputs[w] * self.weigths[w]
            # print summ
        output = unipolar(summ)
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


# and training data
training_data = [[[0, 0, 1], 0], [[0, 1, 1], 0], [[1, 0, 1], 0], [[1, 1, 1], 1]]

training_data_or = [[[0, 0, 1], 0], [[0, 1, 1], 1], [[1, 0, 1], 1], [[1, 1, 1], 1]]

training_data_xor = [[[0, 0, 1], 0], [[0, 1, 1], 1], [[1, 0, 1], 1], [[1, 1, 1], 0]]

training_data_not = [[[0, 1], 1], [[1, 1], 0]]


# print p.guess([0, 0, 1])


def ucz(per, tr_data):
    print tr_data
    # print tr_data[2][1]
    for k in range(50):
        print "numer iteracji" + str(k)
        for tr in tr_data:
            i = tr
            per.train(i[0], i[1])
            print "wagi " + str(per.weigths)
            print ("wejscie " + str(i[0]))
            print " tutaj powinno byc  " + str(i[1]) + " a jest " + str(per.guess(i[0])) +"\n"


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
# pAnd.weigths = [1.005857077587072, 0.45041779087835354, -1.096726188030272]
# pAnd.lr = 0.858687606398
# print pAnd.guess([1, 1, 1])


# ptrainOr = Perceptron()
# ucz(ptrainOr, training_data_or)
# print ptrainOr.lr
# print ptrainOr.weigths

# ptrainOr.lr = 0.167631213692
# ptrainOr.weigths = [0.6586391948364184, 0.43477785666346414, -0.10141305826027047]


# ptrainNot = Perceptron()
# ucz(ptrainNot, training_data_not)
# print ptrainNot.lr
# print ptrainNot.weigths

# xor nie da sie zaprezentwoac jednym perceptronem
# You could also try to change the training sequence in order to model an AND, NOR or NOT function.
# Note that it's not possible to model an XOR function using a single perceptron like this, because the two classes (0 and 1)
# of an XOR function are not linearly separable. In that case you would have to use multiple layers of perceptrons
# (which is basically a small neural network).


# ptrainXor = Perceptron()
# ucz(ptrainXor, training_data_xor)
# print ptrainXor.lr
# print ptrainXor.weigths
