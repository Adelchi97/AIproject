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

    plt.title('iris')
    plt.xlabel('train samples')
    plt.ylabel('error')
    plt.legend()
    plt.show()


def perceptron():
    ds = Dataset()
    data, target = ds.getDataset()

    # for multiclass dataset uses One vs. All by default
    ppt = Perceptron(eta0=0.1, random_state=42)
    numOfSamples = data.shape[0]

    # instantiates the vectors for graphing
    graphDatas = GraphDatas(numOfSamples - 2)
    maxSimulations = 1000

    '''
    X_train, X_test, Y_train, Y_test = train_test_split(data, target,
                                                        test_size=.3,
                                                        random_state=42)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    ppt.fit(X_train_std, Y_train)
    Y_pred = ppt.predict(X_test_std)

    print(Y_test)
    print(Y_pred)

    print(metrics.accuracy_score(Y_pred, Y_test))
'''

    for trainSize in range(2, numOfSamples):

        accuracyToAverage = []
        for simulation in range(maxSimulations):

            X_train, X_test, Y_train, Y_test = train_test_split(data, target,
                                                                test_size=(numOfSamples - trainSize) / numOfSamples,
                                                                random_state=simulation)
            # train the scaler, it standardizes all features to have mean 0 and variance 1
            sc = StandardScaler()
            sc.fit(X_train)
            X_train_std = sc.transform(X_train)
            X_test_std = sc.transform(X_test)

            try:
                # 40 epochs, .1 learning rate
                ppt.fit(X_train_std, Y_train)

                # make prediction
                Y_pred = ppt.predict(X_test_std)
                accuracyToAverage.append(metrics.accuracy_score(Y_test, Y_pred))
            except ValueError:
                simulation = simulation - 1

        accuracy = statistics.mean(accuracyToAverage)
        graphDatas.x_axis[trainSize - 2] = trainSize
        graphDatas.y_axis[trainSize - 2] = 1-accuracy

    # plot results
    plt.plot(graphDatas.x_axis, graphDatas.y_axis, '-o', color='black', markersize=2, linewidth=.5, label='perceptron')


def naiveBayes():
    ds = Dataset()
    data, target = ds.getDataset()

    gnb = GaussianNB()
    numOfSamples = data.shape[0]

    # getting instantiated vector x and y
    graphDatas = GraphDatas(numOfSamples - 1)
    maxSimulations = 100

    # iterates on the number of samples, splitting on every possible division
    for trainSize in range(1, numOfSamples):

        accuracyToAverage = []
        for simulation in range(maxSimulations):
            # random_state set to simulation just to have a different seed for every simulation
            X_train, X_test, Y_train, Y_test = train_test_split(data, target,
                                                                test_size=(numOfSamples - trainSize) / numOfSamples,
                                                                random_state=simulation)
            gnb.fit(X_train, Y_train)
            # predict result of test database
            Y_pred = gnb.predict(X_test)
            # how often is the classifier correct?
            accuracyToAverage.append(metrics.accuracy_score(Y_test, Y_pred))

        accuracy = statistics.mean(accuracyToAverage)
        graphDatas.x_axis[trainSize - 1] = trainSize
        graphDatas.y_axis[trainSize - 1] = 1-accuracy

    # plot results
    plt.plot(graphDatas.x_axis, graphDatas.y_axis, '-o', color='teal', markersize=2, linewidth=.5, label='naive bayes')


if __name__ == "__main__":
    main()
