import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

DECISION_TREE_MAX_DEPTH = 25


def calculate_score(model, X, y):
    """
    Returns a dictionary of statistical information about goodness of prediction
    """
    status_dict = dict()
    prediction = model.predict(X)

    P, N = np.sum(y == 1), np.sum(y == -1)
    TP, TN = np.sum(prediction + y == 2), np.sum(prediction + y == -2)
    FP, FN = np.sum(prediction - y == 2), np.sum(prediction - y == -2)
    accuracy = np.mean(prediction - y == 0)

    status_dict['num_samples'] = len(y)
    status_dict['error'] = 1 - accuracy
    status_dict['accuracy'] = accuracy
    status_dict['FPR'] = FP / max(1, N)
    status_dict['TPR'] = TP / max(1, P)
    status_dict['precision'] = TP / max(1, TP+FP)
    status_dict['specificity'] = TN / max(1, N)

    return status_dict


class Perceptron:
    """
    Class of Perceptron classifier
    """

    def __init__(self):
        """
        Initiate the class
        """
        self.class_name = "Perceptron"
        self.model = None

    def fit(self, X, y):
        """
        Fits and saves model into class variable "model"
        :param X: a design matrix
        :param y: a label vector
        """
        i = 0
        X_shape_rows = X.shape[0]
        X = np.hstack((X, np.ones((X_shape_rows, 1))))
        X_shape_cols = X.shape[1]
        self.model = np.zeros((X_shape_cols))
        while True:
            if (y[i] * (X[i] @ self.model)) > 0:
                break
            else:
                self.model += y[i] * X[i]
                i = np.argmin(y * (X @ self.model))

    def predict(self, X):
        """
        Receive a test design matrix X. Returns a prediction of classifier on X
        :param X: a design matrix
        :return: a prediction of classifier on X
        """
        X_shape = X.shape
        ones = np.ones((X_shape[0], 1))
        return np.sign(np.hstack((X, ones)) @ self.model)

    def score(self, X, y):
        """
        Returns a dictionary of statistical information about goodness of
        prediction to test labels
        :param X: a test set
        :param y: a test set
        :return: a dictionary
        """
        return calculate_score(self, X, y)


class LDA:
    """
    Class of LDA classifier
    """

    def __init__(self):
        """
        Initiate the class
        """
        self.class_name = "LDA"
        self.sigma = None
        self.bias = None
        self.pr = None
        self.mu = None

    def fit(self, X, y):
        """
        Fits and saves model into class variable "model"
        :param X: a design matrix
        :param y: a label vector
        """
        X_y_plus = X[y == 1]
        X_y_minus = X[y == -1]
        y_plus_mean = np.mean(y == 1)
        y_minus_mean = np.mean(y == -1)
        m = len(y)

        self.pr = np.array([y_plus_mean, y_minus_mean])
        self.mu = np.array([np.mean(X_y_plus, axis=0), np.mean(X_y_minus, axis=0)]).transpose()
        try:
            self.sigma = ((X_y_plus - self.mu[:, 0]).transpose() @ (X_y_plus - self.mu[:, 0]) +
                          (X_y_minus - self.mu[:, 1]).transpose() @ (X_y_minus - self.mu[:, 1])) / m
        except ZeroDivisionError:
            print('Unable to train this model')
        self.bias = -0.5 * np.diag(self.mu.transpose() @ np.linalg.inv(self.sigma) @ self.mu)
        self.bias += np.log(self.pr)

    def predict(self, X):
        """
        Receive a test design matrix X. Returns a prediction of classifier on X
        :param X: a design matrix
        :return: a prediction of classifier on X
        """
        predicted_argmax = np.argmax(X @ np.linalg.inv(self.sigma) @ self.mu + self.bias, axis=1)
        predication = (-2 * predicted_argmax)
        predication += 1
        return predication

    def score(self, X, y):
        """
        Returns a dictionary of statistical information about goodness of
        prediction to test labels
        :param X: a test set
        :param y: a test set
        :return: a dictionary
        """
        return calculate_score(self, X, y)


class SVM(SVC):
    """
    Class of SVM classifier
    """

    def __init__(self):
        """
        Initiate the class
        """
        self.name = "SVM"
        self.svc = SVC.__init__(self, C=1e10, kernel='linear')

    def score(self, X, y):
        """
        Returns a dictionary of statistical information about goodness of
        prediction to test labels
        :param X: a test set
        :param y: a test set
        :return: a dictionary
        """
        return calculate_score(self, X, y)


class Logistic(LogisticRegression):
    """
    Class of Logistic Regression classifier
    """

    def __init__(self):
        """
        Initiate the class
        """
        self.name = "Logistic Regression"
        self.logistic_reg = LogisticRegression.__init__(self, solver='liblinear')

    def score(self, X, y):
        """
        Returns a dictionary of statistical information about goodness of
        prediction to test labels
        :param X: a test set
        :param y: a test set
        :return: a dictionary
        """
        return calculate_score(self, X, y)


class DecisionTree(DecisionTreeClassifier):
    """
    Class of Decision tree classifier
    """

    def __init__(self):
        """
        Initiate the class
        """
        self.name = "Decision Tree"
        self.decision_tree = DecisionTreeClassifier.__init__(self,
                                                             max_depth=DECISION_TREE_MAX_DEPTH)

    def score(self, X, y):
        """
        Returns a dictionary of statistical information about goodness of
        prediction to test labels
        :param X: a test set
        :param y: a test set
        :return: a dictionary
        """
        return calculate_score(self, X, y)


if __name__ == "__main__":
    pass