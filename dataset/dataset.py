from numpy import zeros
import numpy as np
from sklearn.preprocessing import LabelEncoder
import csv


class Dataset:

    def __init__(self):
        self.finalDatas = zeros(1)
        self.target = zeros(1)
        return

    def datasetToCsv(self):

        # ********************************************************
        # change this field according to required dataset ********
        file = open("dataset/wine_dataset/wine.data", 'r')
        # ********************************************************

        if file.mode == 'r':
            datas = file.readlines()

            # creates the strucure of the data nparray
            # len(datas) = num of samples, len(datas[0].split(',') = num of attributes+labels
            elaborated_datas = zeros((len(datas), len(datas[0].split(','))))

            # **************************************************
            # change label index according to input datas ******
            label_index = 0
            # **************************************************
            # if targets are strings and not numbers transform them
            if type(datas[0].split(',')[label_index]) == str:
                changed_labels = []
                for line in datas:
                    temp = line.split(',')
                    changed_labels.append(temp[label_index])

                labelEncoder = LabelEncoder()
                changed_labels = labelEncoder.fit_transform(changed_labels)

            i = 0
            for line in datas:
                temp = line.split(',')
                j = 0
                for column in temp:
                    try:
                        elaborated_datas[i][j] = column
                    except ValueError:
                        elaborated_datas[i][j] = changed_labels[i]
                    j = j+1
                i = i+1

            # ***************************************************************
            # change slicing according to the index of the label *********
            self.target = elaborated_datas[:, label_index]
            self.finalDatas = elaborated_datas[:, 1:]
            # ***************************************************************

            # place attributes in right order
            ready_for_csv = zeros((self.target.size, 14))
            ready_for_csv[:, 0] = self.target
            ready_for_csv[:, 1:] = self.finalDatas

            # create csv
            with open('wine.csv', 'w') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerows(ready_for_csv)

    def getDataset(self):

        matrix = np.loadtxt(open("dataset/iris_dataset/iris.csv", "rb"), delimiter=",", skiprows=0)

        self.finalDatas = matrix[:, 1:]
        self.target = matrix[:, 0]

        return self.finalDatas, self.target
