from models import *
from comparison import add_info_to_show
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from time import time

M_SAMPLES = [50, 100, 300, 500]
Q_14_REPEATS_NUMBER = 50


def load_data():
    """
    Returns the loaded data
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_images = np.logical_or((y_train == 0), (y_train == 1))
    test_images = np.logical_or((y_test == 0), (y_test == 1))
    x_train, y_train = x_train[train_images], y_train[train_images]
    x_test, y_test = x_test[test_images], y_test[test_images]
    return x_train, y_train, x_test, y_test


# Question 12
def q_12(X, y): #todo
    """
    The answer for question 12
    Plotting 3 random images of each 0, 1
    :param X: mnist images
    :param y: mnist labels
    """
    zeros = X[y == 0]
    ones = X[y == 1]
    np.random.shuffle(zeros)
    np.random.shuffle(ones)
    drawn = np.vstack((zeros[:3], ones[:3]))
    fig = plt.figure()
    for i in range(6):
        fig.add_subplot(2, 3, i+1)
        plt.imshow(drawn[i])
    plt.savefig("Q12.png")
    plt.show()


def rearrange_data(X):
    """
    The answer for question 13
    :param X: data as a array of size nˆ28ˆ28
    :return: a new matrix of size n X 784
    """
    return np.reshape(X, (-1, 784))


def draw_mnist(X, y, m):  #todo
    """
    Drawing m images and labels from X, y so that there are images with both labels
    :param X: mnist images
    :param y: mnist labels
    :param m: number of samples to draw
    :return: m images and labels
    """
    drawn = np.random.randint(len(y), size=m)
    while np.all(y[drawn] == 0) or np.all(y[drawn] == 1):
        drawn = np.random.randint(len(y), size=m)
    return rearrange_data(X[drawn, :, :]), y[drawn]


# Question 14
def q_14(x_train, y_train, x_test, y_test):  #todo
    """
    Answer to question 14.
    Plotting accuracies and running time of different hypotheses
    :param x_train: images to fit the models
    :param y_train: labels to fit the models
    :param x_test: images to test models
    :param y_test: labels to test models
    """
    # accumulated accuracies
    logistic_accuracy = np.zeros(4)
    SVM_accuracy = np.zeros(4)
    tree_accuracy = np.zeros(4)
    knn_accuracy = np.zeros(4)

    # accumulated time
    running_time = np.zeros((2, 4, 4))

    test_X = rearrange_data(x_test)
    test_y = y_test

    for i in range(Q_14_REPEATS_NUMBER):
        for j, m in enumerate(M_SAMPLES):
            train_X, train_y = draw_mnist(x_train, y_train, m)

            # Fitting models
            logistic = Logistic()
            t = time()
            logistic.fit(train_X, train_y)
            running_time[0, 0, j] += time() - t

            svm = SVM()
            t = time()
            svm.fit(train_X, train_y)
            running_time[0, 1, j] += time() - t

            tree = DecisionTree()
            t = time()
            tree.fit(train_X, train_y)
            running_time[0, 2, j] += time() - t

            knn = KNeighborsClassifier(n_neighbors=50)
            t = time()
            knn.fit(train_X, train_y)
            running_time[0, 3, j] += time() - t

            # Score of models
            t = time()
            logistic_accuracy[j] += logistic.score(test_X, test_y)['accuracy']
            running_time[1, 0, j] += time() - t
            t = time()
            SVM_accuracy[j] += svm.score(test_X, test_y)['accuracy']
            running_time[1, 1, j] += time() - t
            t = time()
            tree_accuracy[j] += tree.score(test_X, test_y)['accuracy']
            running_time[1, 2, j] += time() - t
            t = time()
            knn_accuracy[j] += knn.score(test_X, test_y)
            running_time[1, 3, j] += time() - t

    # Plotting accuracies
    plt.plot(M_SAMPLES, logistic_accuracy / Q_14_REPEATS_NUMBER, label="Logistic Regression", color='tomato', marker='^')
    plt.plot(M_SAMPLES, SVM_accuracy / Q_14_REPEATS_NUMBER, label="SVM", color='cadetblue', marker='^')
    plt.plot(M_SAMPLES, tree_accuracy / Q_14_REPEATS_NUMBER, label="Decision Tree", color='darkseagreen', marker='^')
    plt.plot(M_SAMPLES, knn_accuracy / Q_14_REPEATS_NUMBER, label="K-Nearest Neighbors", color='violet', marker='^')

    add_info_to_show("Q14 - Comparison between accuracies", "m", "accuracy", "Q14-accuracies.png")

    # Plotting average running time
    running_time /= 50
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.plot(M_SAMPLES, running_time[0, 0], label="Logistic Regression", color='tomato', marker='|')
    plt.plot(M_SAMPLES, running_time[0, 1], label="SVM", color='cadetblue', marker='|')
    plt.plot(M_SAMPLES, running_time[0, 2], label="Decision Tree", color='darkseagreen', marker='|')
    plt.plot(M_SAMPLES, running_time[0, 3], label="K-Nearest Neighbors", color='violet', marker='|')

    # add_info_to_show("Training average running time", "m", "average time", "Q14-accuracies.png")

    plt.title("Training average running time")
    plt.xlabel("m")
    plt.ylabel("avg time")
    plt.legend()

    fig.add_subplot(1, 2, 2)
    plt.plot(M_SAMPLES, running_time[1, 0], label="Logistic Regression", color='tomato', marker='|')
    plt.plot(M_SAMPLES, running_time[1, 1], label="SVM", color='cadetblue', marker='|')
    plt.plot(M_SAMPLES, running_time[1, 2], label="Decision Tree", color='darkseagreen', marker='|')
    plt.plot(M_SAMPLES, running_time[1, 3], label="K-Nearest Neighbors", color='violet', marker='|')
    plt.title("Testing average running time")
    plt.xlabel("m")
    plt.ylabel("avg time")
    plt.legend()

    plt.savefig("Q14_running_time.png")
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    q_12(x_train, y_train)
    # q_14(x_train, y_train, x_test, y_test)


