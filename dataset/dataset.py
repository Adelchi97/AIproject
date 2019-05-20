from numpy import zeros
from sklearn.preprocessing import LabelEncoder


class Dataset:

    def __init__(self):
        self.finalDatas = zeros(1)
        self.target = zeros(1)
        return

    def createDataset(self):

        # ********************************************************
        # change this field according to required dataset ********
        file = open("dataset/iris.data", 'r')
        # ********************************************************

        if file.mode == 'r':
            datas = file.readlines()

            # creates the strucure of the data nparray
            # len(datas) = num of samples, len(datas[0].split(',') = num of attributes+labels
            elaborated_datas = zeros((len(datas), len(datas[0].split(','))))

            # **************************************************
            # change label index according to input datas ******
            label_index = len(datas[0].split(','))-1
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
            # change slicing if targets are not in the first column *********
            self.target = elaborated_datas[:, len(elaborated_datas[0])-1]
            self.finalDatas = elaborated_datas[:, :len(elaborated_datas[0])-1]
            # ***************************************************************

        return self.finalDatas, self.target
