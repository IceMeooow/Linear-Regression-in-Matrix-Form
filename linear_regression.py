import numpy as np


class MyLinearRegression:
    '''
    Class implements Linear Regression in Matrix Form (using batch gradient descent).
    '''
    def train(self, train_data, train_labels, iterations=1000, alpha=0.02, progress_messages=True):
        '''
        Method implements the training process of Linear Regression.
        :param train_data: numpy.ndarray, train data, contains only features.
        :param train_labels: numpy.ndarray, train labels for train data.
        :param iterations: int, number of iterations. Default 1000.
        :param alpha: float, learning rate. It controls the step-size in updating the weights. Default 0.02.
        :param progress_messages: bool, whether to print progress messages. Default True

        :return: theta, numpy.ndarray, parameters of the Linear Regression.
        '''
        theta = np.zeros(shape=(train_data.shape[1], 1))
        history = [0] * iterations
        for iteration in range(iterations):
            hypothesis = self.get_hypothesis(train_data, theta)
            gradient = self._get_gradient(train_data, train_labels, hypothesis)
            if (progress_messages is True) and (np.mod(iteration, 1000) == 0):
                print("Iteration {}, sum error = {}".format(iteration, self._cost_function(train_data,
                                                                                           train_labels,
                                                                                           theta)))
            theta = theta - alpha * gradient
            cost = self._cost_function(train_data, train_labels, theta)

            history[iteration] = cost
            # if history[-1] > history[-2]:
            #     break
            # else:
            #     continue

        theta = theta
        return theta, history

    def get_hypothesis(self, data, theta):
        '''
        Method implements the hypothesis  of Linear Regression.
        :param data: numpy.ndarray, numpy.ndarray,  data without labels.
                    The preprocessing of this data must be the same as in train data.
        :param theta: numpy.ndarray, parameters of the Linear Regression.

        :return: hypothesis, numpy.ndarray, predicted labels for data.
        '''
        hypothesis = data.dot(theta)
        return hypothesis

    def _cost_function(self, data, labels, theta):
        '''
        Method implements the cost function of Linear Regression. It is using the function of least squares.
        :param data: numpy.ndarray, data, contains only features.
        :param labels: numpy.ndarray, labels for data.
        :param theta: numpy.ndarray, parameters of the Linear Regression.

        :return: cost, float, the estimate of error.
        '''
        m, _ = labels.shape
        cost = np.sum(1/2 * 1/m * (data.dot(theta) - labels) ** 2)
        return cost

    def _get_gradient(self, data, labels, hypothesis):
        '''
        Method calculates a value of gradient descent (in one step).
        :param data: numpy.ndarray, data, contains only features.
        :param labels: numpy.ndarray, labels for data.
        :param hypothesis: numpy.ndarray, predicted labels for data.

        :return: gradient, numpy.ndarray, numbers.
                It are used for perform update of parameters of the Linear Regression.
        '''
        m, _ = labels.shape
        gradient = 1/m * data.T.dot(hypothesis - labels)
        return gradient
