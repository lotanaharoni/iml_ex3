from models import *
import matplotlib.pyplot as plt

Q_10_REPEATS = 500
M_VALUES = [5, 10, 15, 25, 70]
K = 10000
COLORS_DICT = {'positive': 'blue', 'negative': 'orange', 'Perceptron': 'pink',
               'SVM': 'purple', 'f': 'green', 'LDA': 'gold'}


# Question 8
def draw_points(m):
    """
    Returns m samples drawn from 2-D normal distribution with
    mean [0,0] and unity covariance matrix, and m corresponding labels
    generated from samples using plane vector [0.1, 0.3, -0.5]
    :param m: int
    :return: m samples drawn from 2-D normal distribution
    """
    I = np.eye(2)
    X = np.random.multivariate_normal(np.zeros(2), I, size=m)
    while np.all(np.sign(X @ np.array([0.3, -0.5]) + 0.1) == -1) or \
            np.all(np.sign(X @ np.array([0.3, -0.5]) + 0.1) == 1):
        X = np.random.multivariate_normal(np.zeros(2), I, size=m)
    return X, np.sign(X @ np.array([0.3, -0.5]) + 0.1).astype('int')


def add_info_to_show(title, x_label, y_label, format):
    """
    Receives plot information and add it to plot and then show plot
    """
    plt.title(title)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(format)
    plt.show()


def plot_hyperplane(left, right, left_mult_value, div_value, bias, label):
    """
    Plotting hyperplane
    """
    plt.plot([left, right], [left * left_mult_value / div_value + bias, right * left_mult_value / div_value + bias],
             label=label, color=COLORS_DICT[label])


def plot_hyperplanes(m):
    """
    Plotting hyperplanes of different hypotheses
    :param m: Number of samples to fit the models with
    """
    p = Perceptron()
    svm = SVM()

    X, y = draw_points(m)
    left = np.min(X)
    right = np.max(X)

    p.fit(X, y)
    svm.fit(X, y)

    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='positive', color=COLORS_DICT['positive'])
    plt.scatter(X[y == -1, 0], X[y == -1, 1], label='negative', color=COLORS_DICT['negative'])

    plot_hyperplane(left, right, 0.3, 0.5, 0.1, 'f')
    plot_hyperplane(left, right, p.model[0], -p.model[1], p.model[2] / -p.model[1], 'Perceptron')
    plot_hyperplane(left, right, svm.coef_[0, 0], -svm.coef_[0, 1], svm.intercept_ / -svm.coef_[0, 1], 'SVM')

    add_info_to_show("Q9- Comparison between hyperplanes of hypotheses\nm={0}".format(m), 'x', 'y', "Q9-m={0}.png".format(m))


# Question 9
def q_9():
    """
    The answer for question 9
    """
    for m in M_VALUES:
        plot_hyperplanes(m)


def plot_accuracy(accuracy_by_parameter, label):
    """
    Plot accuracy
    """
    plt.plot(M_VALUES, accuracy_by_parameter / Q_10_REPEATS, label=label, color=COLORS_DICT[label], marker='v')


# Question 10
def q_10():
    """
    The answer for question 10
    """
    accuracy_by_SVM = np.zeros(5)
    accuracy_by_LDA = np.zeros(5)
    accuracy_by_perceptron = np.zeros(5)

    for i in range(Q_10_REPEATS):
        for j, m in enumerate(M_VALUES):
            train_X, train_y = draw_points(m)
            test_X, test_y = draw_points(K)

            p = Perceptron()
            lda = LDA()
            svm = SVM()

            p.fit(train_X, train_y)
            lda.fit(train_X, train_y)
            svm.fit(train_X, train_y)

            accuracy_by_perceptron[j] += p.score(test_X, test_y)['accuracy']
            accuracy_by_LDA[j] += lda.score(test_X, test_y)['accuracy']
            accuracy_by_SVM[j] += svm.score(test_X, test_y)['accuracy']

    plot_accuracy(accuracy_by_perceptron, "Perceptron")
    plot_accuracy(accuracy_by_SVM, "SVM")
    plot_accuracy(accuracy_by_LDA, "LDA")

    add_info_to_show("Q10 - Average Accuracy- Comparison between accuracies of hypotheses", "m", "accuracy", "Q10.png")


if __name__ == '__main__':
    q_9()
    q_10()
