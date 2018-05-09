# --------------------------------------------------------------------------
# -----------------------  Rozpoznawanie Obrazow  --------------------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np
from neuronobj import neuron
from neuronobj import matrixsigmoida


def predict(x):
    """
       Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
       gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
       :param x: macierz o wymiarach NxD
       :return: wektor o wymiarach Nx1
    """
    (n,d)=np.shape(x)
    neuron = pkl.load(open('neuronend.obj', mode='rb'))
    y=np.zeros((n,1))
    for i in range(n):
        all_values = (x[i][:])
        inputs = (np.asfarray(all_values[0:]) / 1 * 0.99) + 0.01
        out = neuron.query(inputs)
        label = np.argmax(out)
        y[i][:]=label

    return (y)


    pass
