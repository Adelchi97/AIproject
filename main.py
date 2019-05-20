def main():
    from sklearn import datasets
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import statistics
    import matplotlib.pyplot as plt

    from graph_model import GraphDatas
    from dataset.dataset import Dataset

    ds = Dataset()
    data, target = ds.createDataset()

    gnb = GaussianNB()
    numOfSamples = data.shape[0]

    # getting instantiated vector x and y
    graphDatas = GraphDatas(numOfSamples - 1)
    maxSimulations = 30

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
        graphDatas.y_axis[trainSize - 1] = accuracy

    # plot results
    plt.plot(graphDatas.x_axis, graphDatas.y_axis, '-o', color='black', markersize=2, linewidth=.5)
    plt.xlabel('train samples')
    plt.ylabel('accuracy')
    plt.title('wine')
    plt.show()


if __name__ == "__main__":
    main()
