from numpy import zeros
from sklearn import preprocessing


class Dataset:

    def __init__(self):
        self.finalDatas = zeros(1)
        self.target = zeros(1)
        return

    def createDataset(self):

        # ********************************************************
        # change this field according to required dataset ********
        file = open("dataset/wine.data", 'r')
        # ********************************************************

        if file.mode == 'r':
            datas = file.readlines()

            # creates the strucure of the data nparray
            # len(datas) = num of samples, len(datas[0].split(',') = num of attributes+labels
            elaborated_datas = zeros((len(datas), len(datas[0].split(','))))

            i = 0
            for line in datas:
                temp = line.split(',')
                j = 0
                for column in temp:
                    elaborated_datas[i][j] = column
                    j = j+1
                i = i+1

            # ***************************************************************
            # change slicing if targets are not in the first column *********
            self.target = elaborated_datas[:, 0]
            self.finalDatas = elaborated_datas[:, 1:]
            # ***************************************************************

        return self.finalDatas, self.target
