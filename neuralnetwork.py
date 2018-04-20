kuy = kuy.ravel()

import arff 
import numpy as np
from preprocessing import preprocessing,getoutputData
np.set_printoptions(threshold='nan')

dataset = arff.load(open('Autism-Child-Data.arff', 'rb'))
data = np.array(dataset['data'])
X = preprocessing(data,21)
y = np.array(getoutputData(X))

X = np.amax(X, axis=0)

y = y/100

class layer(object) :
    def __init__(self,inputSize,outputSize,hiddenSize) :
        self.inputSize = 18
        self.outputSize = 2
        self.hiddenSize = 3
        self.w1 = np.random.randn(1, 3) 
        self.w2 = np.random.randn(1, 3)
    def forward(self, X):
        self.z = np.dot(X, self.w1) 
        self.z2 = 1/(1+np.exp(self.w1))
        self.z3 = np.dot(self.z2, self.w2)
        o = 1/(1+np.exp(self.z3))
        return o
    def backward(self, X, y, o):
        self.o_error = y - o # error in output
        self.o_delta = self.o_error*( o*(1-o)) # applying derivative of sigmoid to error
        self.z2_error = self.o_delta.dot(self.w2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*(self.z2*(1-self.z2)) # applying derivative of sigmoid to z2 error

        self.w1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        self.w2 += self.z2.T.dot(self.o_delta)
    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)



if __name__ == "__main__" :
    NN = layer(18,2,3)
    for i in xrange(10000): # trains the NN 1,000 times
        print " #" + str(i) + "\n"
        print "Input (scaled): \n" + str(X)
        print "Actual Output: \n" + str(y)
        print "Predicted Output: \n" + str(NN.forward(X))
        print "Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))) # mean sum squared loss
        print "\n"
        NN.train(X, y)   


