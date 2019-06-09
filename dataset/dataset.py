from numpy import zeros
import numpy as np
from sklearn.preprocessing import LabelEncoder
import csv

import pandas as pd


class Dataset:

    def __init__(self):
        self.finalData = zeros(1)
        self.target = zeros(1)
        return

    def datasetToCsv(self):

        # ********************************************************
        # change this field according to required dataset ********
        file = open("ecoli_dataset/ecoli.data", 'r')
        # ********************************************************

        if file.mode == 'r':
            datas = file.readlines()

            # add datas to a list
            data_list = []
            for line in datas:
                inside_list = []
                data_list.append(inside_list)
                temp = line.split()
                for column in temp:
                    inside_list.append(column)

            dataframe = pd.DataFrame(data_list)
            datas_in_number = dataframe.apply(LabelEncoder().fit_transform)
            elaborated_datas = np.array(datas_in_number)

            # **************************************************
            # change label index according to input data ******
            label_index = elaborated_datas.shape[1]-1
            # **************************************************

            # ***************************************************************
            # change slicing according to the index of the label *********
            self.target = elaborated_datas[:, label_index]
            self.finalData = elaborated_datas[:, :label_index]
            # ***************************************************************

            # place attributes in right order
            ready_for_csv = zeros((self.target.size, elaborated_datas.shape[1]-1))
            ready_for_csv[:, 1:] = self.finalData[:, 1:]
            ready_for_csv[:, 0] = self.target

            # create csv
            with open('ecoli.csv', 'w') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerows(ready_for_csv)

    def getDataset(self):

        matrix = np.loadtxt(open("dataset/ecoli_dataset/ecoli.csv", "rb"), delimiter=",", skiprows=0)

        self.finalData = matrix[:, 1:]
        self.target = matrix[:, 0]

        return self.finalData, self.target


def main():
    dataset = Dataset()
    dataset.datasetToCsv()


if __name__ == "__main__":
    main()
