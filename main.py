from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statistics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

from graph_model import GraphDatas
from dataset.dataset import Dataset


def main():
    naiveBayes()
    perceptron()

    plt.title('pima')
    plt.xlabel('train samples')
    plt.ylabel('error')
    plt.legend()
    plt.show()


def perceptron():
    print("beginning perceptron")
    ds = Dataset()
    data, target = ds.getDataset()

    # for multiclass dataset uses One vs. All by default
    ppt = Perceptron(eta0=0.1, random_state=42)
    numOfSamples = data.shape[0]

    # instantiates the vectors for graphing
    graphDatas = GraphDatas(numOfSamples-10-100)
    maxSimulations = 500

    i = 0
    for trainSize in range(10, numOfSamples-100, 1):
        print("train size: ", trainSize)
        accuracyToAverage = []
        for simulation in range(maxSimulations):

            X_train, X_test, Y_train, Y_test = train_test_split(data, target,
                                                                test_size=(numOfSamples - trainSize) / numOfSamples,
                                                                random_state=None)
            # train the scaler, it standardizes all features to have mean 0 and variance 1
            sc = StandardScaler()
            sc.fit(X_train)
            X_train_std = sc.transform(X_train)
            X_test_std = sc.transform(X_test)

            try:
                ppt.fit(X_train_std, Y_train)
                # make prediction
                Y_pred = ppt.predict(X_test_std)
                accuracyToAverage.append(metrics.accuracy_score(Y_test, Y_pred))
            except ValueError:
                simulation = simulation - 1

        accuracy = statistics.mean(accuracyToAverage)
        try:
            graphDatas.x_axis[i] = trainSize
            graphDatas.y_axis[i] = 1-accuracy
        except IndexError:
            print("index out")
        i = i + 1

    # plot results
    plt.plot(graphDatas.x_axis, graphDatas.y_axis, '-o', color='black', markersize=2, linewidth=.5, label='perceptron')


def naiveBayes():
    print("beginning naive bayes")
    ds = Dataset()
    data, target = ds.getDataset()

    gnb = GaussianNB()
    numOfSamples = data.shape[0]

    # getting instantiated vector x and y
    graphDatas = GraphDatas(numOfSamples-10-100)
    maxSimulations = 500

    # iterates on the number of samples, splitting data for each try
    i=0
    for trainSize in range(10, numOfSamples-100, 1):
        print("train size: ", trainSize)
        accuracyToAverage = []
        for simulation in range(maxSimulations):
            X_train, X_test, Y_train, Y_test = train_test_split(data, target,
                                                                test_size=(numOfSamples - trainSize) / numOfSamples,
                                                                random_state=None)
            sc = StandardScaler()
            sc.fit(X_train)
            X_train_std = sc.transform(X_train)
            X_test_std = sc.transform(X_test)

            gnb.fit(X_train_std, Y_train)
            # predict result of test database
            Y_pred = gnb.predict(X_test_std)
            # how often is the classifier correct?
            accuracyToAverage.append(metrics.accuracy_score(Y_test, Y_pred))

        accuracy = statistics.mean(accuracyToAverage)
        try:
            graphDatas.x_axis[i] = trainSize
            graphDatas.y_axis[i] = 1-accuracy
        except IndexError:
            print("andato fuori")
        i = i + 1

    # plot results
    plt.plot(graphDatas.x_axis, graphDatas.y_axis, '-o', color='teal', markersize=2, linewidth=.5, label='naive bayes')


if __name__ == "__main__":
    main()
