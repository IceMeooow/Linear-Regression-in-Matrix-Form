import numpy as np
import random


class TrainTestBreaker:
    '''
    Class is dedicated for splitting data/labels into train and test subsets.
    '''
    def __init__(self, data, labels):
        '''
        Initial method for initializing instance.
        :param data: data, numpy.ndarray, which contains only features.
        :param labels: numpy.ndarray, labels for data.
        '''
        self.data = data
        self.labels = labels

    def split_data_on_train_test_set(self, size_of_train_data_in_percent, random_state):
        '''
        Method splits (randomly) data, labels in train/test sets.
        :param size_of_train_data_in_percent:  float, should be between 0 and 100 and
                                            represent the proportion of the data to include in the train split.
                random_state: int or None, it is the seed used by the random number generator

        :return: train_data: numpy.ndarray, train data, contains only features.
                test_data:  numpy.ndarray, test data, contains only features.
                train_labels: numpy.ndarray, train labels, contains only labels.
                test_labels: numpy.ndarray, test labels, contains only labels.
        '''
        row, _ = self.data.shape
        size = round(row * size_of_train_data_in_percent / 100)
        data_index = np.arange(row).tolist()
        random.seed(random_state)
        train_index_list = random.sample(data_index, size)
        test_index_list = [item for item in data_index if item not in train_index_list]
        train_data = self.data[train_index_list, :]
        test_data = self.data[test_index_list, :]
        train_labels = self.labels[train_index_list]
        test_labels = self.labels[test_index_list]
        return train_data, test_data, train_labels, test_labels
