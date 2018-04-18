import arff 
import numpy as np
import pprint

dataset = arff.load(open('Autism-Child-Data.arff', 'rb'))
data = np.array(dataset['data'])


def preprocessing(data_arff_file,number_of_attributes) :
    for i in range(len(data_arff_file)) :
        for j in range(number_of_attributes) :
            print data_arff_file[i,j]

        break

a = np.empty((0,3), int)
        

print a