import arff 
import numpy as np
import pprint

dataset = arff.load(open('Autism-Child-Data.arff', 'rb'))
data = np.array(dataset['data'])


def preprocessing(data_arff_file,number_of_attributes) :
    Done = np.empty((0,number_of_attributes), int)
    temp =0
    for x in range(len(data_arff_file)) :
        newdataset = []
        for y in range(number_of_attributes) :   
            newdataset.append([data_arff_file[x,y]])
            
        Done = np.append(Done,[newdataset])
        
    return Done


print  preprocessing(data,21)