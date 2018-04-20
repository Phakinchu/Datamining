import arff 
import numpy as np
import pprint
np.set_printoptions(threshold='nan')

def preprocessing(data_arff_file,number_of_attributes) :
    new_output_dataset = [np.empty((1,number_of_attributes), int)]
    new_output_datasettemp = []
    temp2 = 0
    for i in range(len(data_arff_file)) :
        newdataset= []
        for j in range(number_of_attributes) :
            if(j == 10) :
                if(data_arff_file[i,j] == None) :
                    data_arff_file[i,j] = float(6)
                data_arff_file[i,j] = float(((data_arff_file[i,j]-1)/(11-4))*(1-0))
            elif(j == 11) :
                if(data_arff_file[i,j] == "f") :
                    data_arff_file[i,j] = float(0)
                else :
                    data_arff_file[i,j] = 1
            elif(j == 0|j == 1 | j== 2 | j==3 | j== 4 | j==5 |j == 6 | j==7 | j==8 | j==9) :
                if(data_arff_file[i,j] == "0") :
                    data_arff_file[i,j] = float(0)
                elif(data_arff_file[i,j] == "1") :
                    data_arff_file[i,j] = float(1)
            elif(j == 12) :
                if(data_arff_file[i,j] == None) :
                    data_arff_file[i,j] = float(0)
                else :
                    data_arff_file[i,j] = float(1)
            elif(j == 13) :
                if(data_arff_file[i,j] == "no") :
                    data_arff_file[i,j] = float(0)
                else :
                    data_arff_file[i,j] = float(1)
            elif(j == 14) :
                if(data_arff_file[i,j] == "no") :
                    data_arff_file[i,j] = float(0)
                else :
                    data_arff_file[i,j] = float(1)
            elif(j == 15) :
                if(data_arff_file[i,j] == "Jordan") :
                    data_arff_file[i,j] = float(0)
                else :
                    data_arff_file[i,j] = float(1)
            elif(j == 16) :
                if(data_arff_file[i,j] == "no") :
                    data_arff_file[i,j] = float(0)
                else :
                    data_arff_file[i,j] = float(1)
            elif(j == 18) :
                if(data_arff_file[i,j] == "'4-11 years'") :
                    data_arff_file[i,j] = float(0)
                else :
                    data_arff_file[i,j] = float(1)
            elif(j == 17) :
                if(data_arff_file[i,j] == None) :
                    data_arff_file[i,j] = float(6)
                data_arff_file[i,j] = ((data_arff_file[i,j]-1)/(10-1))*(1-0)       
            elif(j == 19) :
                if(data_arff_file[i,j] == None) :
                    data_arff_file[i,j] = "Parent"
                if(data_arff_file[i,j] == "Parent") :
                    data_arff_file[i,j] = float(0)
                elif(data_arff_file[i,j] == "Self") :
                    data_arff_file[i,j] = float(0.33)
                elif(data_arff_file[i,j] == "Relative") :
                    data_arff_file[i,j] = float(0.67)
                else :
                    data_arff_file[i,j] = float(1)
            elif(j == 20) :
                if(data_arff_file[i,j] == "NO") :
                    data_arff_file[i,j] = float(0)
                else :
                    data_arff_file[i,j] = float(1)

            newdataset.append([data_arff_file[i,j]])
        new_output_datasettemp = newdataset
        # print "Dontemp = ",new_output_datasettemp
        new_output_dataset = np.append(new_output_dataset,new_output_datasettemp)
        new_output_dataset = new_output_dataset.astype(float)
        temp2 += 1

    new_output_dataset = np.reshape(new_output_dataset,(temp2+1,number_of_attributes))
    new_output_dataset= np.delete(new_output_dataset,[0],0)
    new_output_dataset= np.delete(new_output_dataset,12,1)
    new_output_dataset= np.delete(new_output_dataset,14,1)
    new_output_dataset= np.delete(new_output_dataset,16,1)
    return new_output_dataset

def train(new_output_dataset) :
    # new_output_dataset = np.delete(new_output_dataset,17,1)
    train = np.copy(new_output_dataset)
    train= np.delete(train,17,1)
    for i in range(92) :
        train= np.delete(train,[291-(i)], 0)
    return train

def test(new_output_dataset) :
    test = np.copy(new_output_dataset)
    test= np.delete(test,17,1)
    for i in range(200) :
        test= np.delete(test,[291-(i)], 0)
    return test

def getoutputData(new_output_dataset,trainortest) :
    kuy = new_output_dataset
    for i in range(trainortest) :
        kuy= np.delete(kuy,[291-(i)], 0)
    kuy= np.delete(kuy,0,1)
    kuy= np.delete(kuy,0,1)
    kuy= np.delete(kuy,0,1)
    kuy= np.delete(kuy,0,1)
    kuy= np.delete(kuy,0,1)
    kuy= np.delete(kuy,0,1)
    kuy= np.delete(kuy,0,1)
    kuy= np.delete(kuy,0,1)
    kuy= np.delete(kuy,0,1)
    kuy= np.delete(kuy,0,1)
    kuy= np.delete(kuy,0,1)
    kuy= np.delete(kuy,0,1)
    kuy= np.delete(kuy,0,1)
    kuy= np.delete(kuy,0,1)
    kuy= np.delete(kuy,0,1)
    kuy= np.delete(kuy,0,1)
    kuy= np.delete(kuy,0,1)
    return kuy

if __name__ == "__main__" :
    dataset = arff.load(open('Autism-Child-Data.arff', 'rb'))
    data = np.array(dataset['data'])
    X = preprocessing(data,21)
    y = train(X)
    z = test(X)
    outputtrain = getoutputData(X,10)
    outputtest = getoutputData(X,282)
    print len(outputtest)
