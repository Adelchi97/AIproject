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
        file = open("ionosphere_dataset/ionosphere.data", 'r')
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

            df = pd.DataFrame(data_list)
            le = LabelEncoder()
            df[34] = le.fit_transform(df[34])
            # data_in_number = df.apply(LabelEncoder().fit_transform)

            elaborated_data = np.array(df)

            # **************************************************
            # change label index according to input data ******
            label_index = elaborated_data.shape[1] - 1
            # **************************************************

            # ***************************************************************
            # change slicing according to the index of the label *********
            self.target = elaborated_data[:, label_index]
            self.finalData = elaborated_data[:, :label_index]
            # ***************************************************************

            # place attributes in right order
            ready_for_csv = zeros((self.target.size, elaborated_data.shape[1]))
            ready_for_csv[:, 1:] = self.finalData
            ready_for_csv[:, 0] = self.target

            # create csv
            with open('ionosphere.csv', 'w') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerows(ready_for_csv)

    def getDataset(self):

        matrix = np.loadtxt(open("dataset/adult_dataset/adult.csv", "rb"), delimiter=",", skiprows=0)

        self.finalData = matrix[: , 1:]
        self.target = matrix[:, 0]

        return self.finalData, self.target


def main():
    dataset = Dataset()
    dataset.datasetToCsv()


if __name__ == "__main__":
    main()
