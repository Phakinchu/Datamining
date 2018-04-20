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
o = getoutputData(x,73)

l = test(x)
u = getoutputData(x,219)

def entropy() :
    kuy = math.log2(10)
    return 0