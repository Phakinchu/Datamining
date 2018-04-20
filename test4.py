import arff 
import numpy as np
import math
from preprocessing import preprocessing,getoutputData,test,train

dataset = arff.load(open('Autism-Child-Data.arff', 'rb'))
data = np.array(dataset['data'])
#Input array
x = preprocessing(data,21)
#Output
y = train(x)
o = getoutputData(x,73)


def Sigmoid(x):
    return 1/(1 + np.exp(-x))

def derivative_sigmoid(x):  
    return  x * (1-x)

def trainn(inputdata,outputdata):
    epoch = 5000
    lr = 0.01
    win = np.random.uniform(size = (7,3))
    biin = np.random.uniform(size = (1,3))
    wout = np.random.uniform(size = (3,3))
    biout = np.random.uniform(size= (1,3))

    for i in range(epoch):
        hidden_input = np.dot(inputdata,win)
        hidden_layer = hidden_input + biin
        hidden_activate = Sigmoid(hidden_layer)
        output_input = np.dot(hidden_activate,wout)
        output_layer = output_input + biout
        output = Sigmoid(output_layer)

        E =  outputdata - output
        slope_output = derivative_sigmoid(output)
        slope_hidden = derivative_sigmoid(hidden_activate)
        d_output = E * slope_output
        Error_hidden = d_output.dot(wout.T)
        d_hidden = Error_hidden * slope_hidden
        wout += hidden_activate.T.dot(d_output)*lr
        biout += np.sum(d_output, axis=0 ,keepdims=True)*lr
        win  += inputdata.T.dot(d_hidden)*lr
        biin += np.sum(d_hidden,axis=0,keepdims=True)*lr

    return wout,biout,win,biin

def testt(wout,biout,win,biin,inputdata):
    
    hidden_input = np.dot(inputdata,win)
    hidden_layer = hidden_input + biin
    hidden_activate = Sigmoid(hidden_layer)
    output_input = np.dot(hidden_activate,wout)
    output_layer = output_input + biout
    output = Sigmoid(output_layer)

    x = np.around(output)
    roundx = x.astype(int)
    return roundx
            
def checkerror(target,real):
    return (target == real).sum()/float(target.size)*100

def main():
    for kfold in range(10):
        inputdata,outputdata = Readcsv()
        trainin,trainout,testin,testout = fold(inputdata,outputdata)
        data = np.array(inputdata)
        out = np.array(outputdata)
        wout,biout,win,biin = train(trainin,trainout)
        target = test(wout,biout,win,biin,testin)
        print(checkerror(target,testout))

if __name__ == '__main__':
    main()