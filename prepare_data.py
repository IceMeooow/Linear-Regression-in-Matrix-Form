import numpy as np


class ReadAndPrepareData:
    '''
    Class is dedicated for reading and processing of data.
    '''
    def __init__(self, path):
        '''
        Initial method for initializing instance.
        :param path: path to file in csv format.
        '''
        self.path = path

    def do_preparation(self):
        '''
        Method reads data from source.Then implements  preprocessing of data:
        - split data on data (contains only features) set  and labels set,
        - normalize data,
        - add unit column to data.

        :return: data: numpy.ndarray, contains only features.
                labels: numpy.ndarray, labels for data.
        '''
        dataset = self._read_dataset()
        data, labels = self._select_data_and_labels(dataset)
        data = self._normalize_data(data)
        data = self._add_column_with_one(data)
        return data, labels

    def _read_dataset(self):
        '''
        Method performs reading data from csv file and converts it to numpy.ndarray.
        :return: dataset in the format of numpy.ndarray.
        '''
        dataset = np.genfromtxt(self.path, delimiter=',', skip_header=1)
        return dataset

    def _select_data_and_labels(self, dataset):
        '''
        Method selects the right columns for data (contains only features) and labels.
        :param dataset: numpy.ndarray, dataset which needs split.
        :return: data: numpy.ndarray, contains only features.
                labels: numpy.ndarray, labels for data.
        '''
        data = dataset[:, :8]
        labels = dataset[:, 8:]
        return data, labels

    def _normalize_data(self, data):
        '''
        Method scale input vectors individually to unit norm (norm in  l2).
        :param data: numpy.ndarray, input data.
        :return: data, numpy.ndarray, data are normalized.
        '''
        norma = np.sqrt(np.sum(np.abs(data) ** 2, axis=0))
        data = data / norma
        return data

    def _add_column_with_one(self, data):
        '''
        Method adds unit column to data.
        :param data:  numpy.ndarray, input data.
        :return: data, numpy.ndarray, data with additional column.
        '''
        ones_col = np.ones(shape=(data.shape[0], 1))
        data = np.hstack((data, ones_col))
        return data
