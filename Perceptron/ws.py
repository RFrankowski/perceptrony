import unittest
import pickle
import numpy as np
import sys
from scipy import special
from neuronobj import neuron
from neuronobj import matrixsigmoida
import scipy
from scipy import ndimage
import os

def cls():
    os.system('cls' if os.name=='nt' else 'clear')






#with open('neuron','wb')as file: pickle.dump(neuron,protocol=pickle.HIGHEST_PROTOCOL)

TEST_DATA = pickle.load(open('train.pkl', mode='rb'))


img=TEST_DATA[0][0]
a=max(TEST_DATA[1])
tt=neuron(3136,200,36,0.1)
(a,b)=np.shape(TEST_DATA[0])
for j in range(8):
    for i in range(0,a):
        all_values=(TEST_DATA[1][i],TEST_DATA[0][i])
        inputs=(np.asfarray(all_values[1:])/1*0.99)+0.01
        targets=np.zeros(36)+0.01
        targets[int(all_values[0])]=0.99
        tt.train(inputs,targets)

    for i in range(0, a):
        all_values=(TEST_DATA[1][i],TEST_DATA[0][i])
        shape = np.reshape(TEST_DATA[0][i], (56, 56))
        shape2 = scipy.ndimage.interpolation.rotate(shape, 1, reshape=False)
        shapen2 = scipy.ndimage.interpolation.rotate(shape, -1, reshape=False)
        shape2 = np.reshape(shape2, ((56 * 56), 1))
        shape2 = shape2.squeeze()
        shapen2 = np.reshape(shapen2, ((56 * 56), 1))
        shapen2 = shapen2.squeeze()
        all_values2=(TEST_DATA[1][i],shape2)
        all_values3=(TEST_DATA[1][i],shapen2)
        inputs2=(np.asfarray(all_values2[1:])/1*0.99)+0.01
        inputs3=(np.asfarray(all_values3[1:])/1*0.99)+0.01
        targets[int(all_values[0])]=0.99
        tt.train(inputs2, targets)
        tt.train(inputs3, targets)

    print(j)






with open('neuronend5.obj','wb') as output:
    pickle.dump(tt,output ,protocol=pickle.HIGHEST_PROTOCOL)


#lj1886
#228315

