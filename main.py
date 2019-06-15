from sklearn.naive_bayes import MultinomialNB
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

    plt.title('adult')
    plt.xlabel('train samples')
    plt.ylabel('error')
    plt.legend()
    plt.show()


def perceptron():
    print("beginning perceptron")
    ds = Dataset()
    data, target = ds.getDataset()

    # for multiclass dataset uses One vs. All by default
    ppt = Perceptron(eta0=0.1, random_state=42, max_iter=100)
    num_of_samples = data.shape[0]

    # instantiates the vectors for graphing
    graph_datas = GraphDatas(num_of_samples - 10-32400)
    max_simulations = 1000

    i = 0
    for trainSize in range(10, num_of_samples-32400, 1):
        print("train size: ", trainSize)
        accuracy_to_average = []
        for simulation in range(max_simulations):

            x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                                test_size=(num_of_samples - trainSize) / num_of_samples,
                                                                random_state=None)
            # train the scaler, it standardizes all features to have mean 0 and variance 1
            sc = StandardScaler()
            sc.fit(x_train)
            x_train_std = sc.transform(x_train)
            x_test_std = sc.transform(x_test)

            try:
                ppt.fit(x_train_std, y_train)
                # make prediction
                y_pred = ppt.predict(x_test_std)
                accuracy_to_average.append(metrics.accuracy_score(y_test, y_pred))
            except ValueError:
                simulation = simulation - 1

        accuracy = statistics.mean(accuracy_to_average)
        try:
            graph_datas.x_axis[i] = trainSize
            graph_datas.y_axis[i] = 1 - accuracy
        except IndexError:
            print("index out")
        i = i + 1

    # plot results
    plt.plot(graph_datas.x_axis, graph_datas.y_axis, '-o', color='black', markersize=2, linewidth=.5, label='perceptron')


def naiveBayes():
    print("beginning naive bayes")
    ds = Dataset()
    data, target = ds.getDataset()

    # var_smoothing Portion of the largest variance of all features that is added to variances for calculation stability
    gnb = MultinomialNB(alpha=2)  # var_smoothing=1e-01)
    num_of_samples = data.shape[0]

    # getting instantiated vector x and y
    graph_data = GraphDatas(num_of_samples - 10-32400)
    max_simulations = 1000

    # iterates on the number of samples, splitting data for each try
    i = 0
    for trainSize in range(10, num_of_samples-32400, 1):
        print("train size: ", trainSize)
        accuracy_to_average = []
        for simulation in range(max_simulations):
            x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                                test_size=(num_of_samples - trainSize) / num_of_samples,
                                                                random_state=None)

            gnb.fit(x_train, y_train)
            # predict result of test database
            y_pred = gnb.predict(x_test)
            # how often is the classifier correct?
            accuracy_to_average.append(metrics.accuracy_score(y_test, y_pred))

        accuracy = statistics.mean(accuracy_to_average)
        try:
            graph_data.x_axis[i] = trainSize
            graph_data.y_axis[i] = 1 - accuracy
        except IndexError:
            print("index out")
        i = i + 1

    # plot results
    plt.plot(graph_data.x_axis, graph_data.y_axis, '-o', color='teal', markersize=2, linewidth=.5, label='naive bayes')


if __name__ == "__main__":
    main()
