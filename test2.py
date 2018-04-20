import arff 
import numpy as np
import math
from preprocessing import preprocessing,getoutputData,test,train
np.set_printoptions(threshold='nan')

dataset = arff.load(open('Autism-Child-Data.arff', 'rb'))
data = np.array(dataset['data'])
X = preprocessing(data,21)
def sigmoid(x,deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

y = train(X)
# y = [[float(n) for n in m] for m in y]
o = getoutputData(X,10)
z = test(X)
p = getoutputData(z,282)
weight_input = 2*np.random.random((17,5)) - 1
weight_hidden = 2*np.random.random((6,1)) - 1

print o
learning_rate = [[0.01]]

print "Start : "

# code

for i in xrange(1,1500,1):
    for j in xrange(282):
        input_layer = np.array([y[j]])
        hidden_layer_out = sigmoid(np.dot(input_layer, weight_input))
        hidden_layer_out = np.insert(hidden_layer_out, 5, 1, axis=1)
        output_layer = sigmoid(np.dot(hidden_layer_out, weight_hidden))
        output_error = o[j] - output_layer
        output_gradient = output_error * sigmoid(output_layer, deriv=True)
        hidden_layer_error = output_gradient.dot(weight_hidden.T)
        hidden_gradient = hidden_layer_error * sigmoid(hidden_layer_out, deriv=True)
        hidden_gradient = np.delete(hidden_gradient, 5, 1)
        weight_hidden += (hidden_layer_out.T.dot(output_gradient) * learning_rate)
        weight_input += (input_layer.T.dot(hidden_gradient) * learning_rate)
        if (i % 10 == 0 and j == 281) :
            #error
            print "Error:" + str(0.5*(np.square(output_error)))
            print "Weight of input layer"
            print weight_input
            print "Weight of hidden layer"
            print weight_hidden
            print "Output layer"
            print output_layer



