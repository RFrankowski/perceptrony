
import pickle as pkl
import numpy as np
from neuronobj import neuron
from neuronobj import matrixsigmoida
from predict import  predict
import scipy
from scipy import nd

"""
TEST_DATA = pkl.load(open('train.pkl', mode='rb'))

X_train=TEST_DATA[0]
expected=TEST_DATA[1]
score=[]




(a,b)=np.shape(TEST_DATA[0])
a=int(a/4)
X_test=X_train[0:a]
prit=X_test[0]
prit=np.reshape(prit,(56,56))
y=predict(X_test)

for i in range(0,a):
    if y[i]==expected[i]:
        score.append(1)
    else:
        score.append(0)
pkt=sum(score)
pkt=(pkt/a)*100
#1~71%
#2~95%
#3~93$


print(pkt)
"""
neuron = pkl.load(open('neuronend2.obj', mode='rb'))
print(neuron.who)
print(neuron.wih)
