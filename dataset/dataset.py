from numpy import zeros
import numpy as np
from sklearn.preprocessing import LabelEncoder
import csv

import pandas as pd


class Dataset:

    def __init__(self):
        self.finalDatas = zeros(1)
        self.target = zeros(1)
        return

    def datasetToCsv(self):

        # ********************************************************
        # change this field according to required dataset ********
        file = open("breastCancer_dataset/breastCancer.data", 'r')
        # ********************************************************

        if file.mode == 'r':
            datas = file.readlines()

            # add datas to a list
            data_list = []
            for line in datas:
                inside_list = []
                data_list.append(inside_list)
                temp = line.split(',')
                for column in temp:
                    inside_list.append(column)

            dataframe = pd.DataFrame(data_list)
            datas_in_number = dataframe.apply(LabelEncoder().fit_transform)
            elaborated_datas = np.array(datas_in_number)
            #elaborated_datas = np.array(data_list)

            # **************************************************
            # change label index according to input datas ******
            label_index = elaborated_datas.shape[1]-1
            # **************************************************

            # ***************************************************************
            # change slicing according to the index of the label *********
            self.target = elaborated_datas[:, label_index]
            self.finalDatas = elaborated_datas[:, 1:label_index]
            # ***************************************************************

            # place attributes in right order
            ready_for_csv = zeros((self.target.size, 10))
            ready_for_csv[:, 1:] = self.finalDatas
            ready_for_csv[:, 0] = self.target

            # create csv
            with open('breastCancer.csv', 'w') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerows(ready_for_csv)

    def getDataset(self):

        matrix = np.loadtxt(open("dataset/breastCancer_dataset/breastCancer.csv", "rb"), delimiter=",", skiprows=0)

        self.finalDatas = matrix[:, 1:]
        self.target = matrix[:, 0]

        return self.finalDatas, self.target


def main():
    dataset = Dataset()
    dataset.datasetToCsv()


if __name__ == "__main__":
    main()
