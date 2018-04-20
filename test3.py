import arff 
import numpy as np
import math
import random
from preprocessing import preprocessing,getoutputData,test,train


dataset = arff.load(open('Autism-Child-Data.arff', 'rb'))
data = np.array(dataset['data'])

#Input array
x = preprocessing(data,21)
#Output
y = train(x)
o = getoutputData(x,92)

l = test(x)
u = getoutputData(x,200)

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Variable initialization
lr=0.01 #Setting learning rate
inputlayer_neurons = y.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 2 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(17,3))
bh=np.random.uniform(size=(1,3))
wout=np.random.uniform(size=(3,2))
bout=np.random.uniform(size=(1,2))

for i in range(100):
    #Forward Propogation
    hidden_layer_input1=np.dot(y,wh)
    hidden_layer_input=hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1+ bout
    output = sigmoid(output_layer_input)

    #Backpropagation
    E = o-output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hiddenlayer_activations.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += y.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
    # print "new wieght  = ",wout
    # print "new bias  = ",bh
    # print "output = ",output

def testfunction(wh,bh,wout,bout,l,u) :
    answer = 0
    for i in range(73) :
        check = 0
        hidden_layer_input1=np.dot(l,wh)
        hidden_layer_input=hidden_layer_input1 + bh
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1=np.dot(hiddenlayer_activations,wout)
        output_layer_input= output_layer_input1+ bout
        output = sigmoid(output_layer_input)
        g = np.around(output)
        roundx = g.astype(int)
        # if( output[i,0] > output[i,1] ) :
        #     check = 1
        #     # 0 1 = 0
        # elif( output[i,0] < output[i,1] ) :
        #     check = 0
        #     # 1 0 = 1
        # # print " i = ",i,"and checkanswer ", checkanswer[i]
        # if( check == u[i] ) :
        #     # print "u[",i,"] = " , u[i]
        #     answer += 1

    # print "test out put is",(answer*73)/100,"%"
    print roundx
    return roundx

def checkerror(target,real):
    return (target == real).sum()/float(target.size)*100



hi = testfunction(wh,bh,wout,bout,l,u)
hi2 = checkerror(hi,u)
print hi
print "kuy",hi2
