from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score
import config


class ModelEvaluator:
    """
    Used to evaluate weights on test dataset. Evaluation is performed with sklearn due to difficulties of
    changing weights for pyspark's logistic regression.
    """

    def __init__(self, X_test, Y_test):
        """
        Creates a logistic regression object whose weights will be overriden.
        :param X_test: numpy array of test inputs
        :param Y_test: numpy array of test labels
        """
        self.X_test = X_test
        self.Y_test = Y_test
        self.logisticRegr = LogisticRegression()
        # TODO: remove
        # print(self.X_test)
        # print(self.Y_test)
        # for i in Y_test:
        #     if i == 0:
        #         print('y-------------------------:', 'we have zero')
        #         break

        self.logisticRegr.fit(self.X_test, self.Y_test)

    def accuracy(self, weights, intercepts):
        """
        Calculates accuracy on test dataset given new weights and intercepts
        :param weights: numpy array of weights
        :param intercepts: numpy array of intercepts
        :return: returns accuracy on test dataset
        """
        self.logisticRegr.coef_ = weights  # override weights and coefficients
        self.logisticRegr.intercept_ = intercepts

        if config.F1_SCORE:
            y_pred = self.logisticRegr.predict(self.X_test)
            return f1_score(self.Y_test, y_pred, average='macro')

        return self.logisticRegr.score(self.X_test, self.Y_test)
        # return self.logisticRegr.score(self.X_test, self.Y_test)
