from prepare_data import ReadAndPrepareData
from train_test_split import TrainTestBreaker
from linear_regression import MyLinearRegression
from sklearn.metrics import r2_score


data, labels = ReadAndPrepareData("prices.csv").do_preparation()
train_data, test_data, train_labels, test_labels = TrainTestBreaker(data, labels).split_data_on_train_test_set(70, 1)

# Training process
theta, history = MyLinearRegression().train(train_data, train_labels, 500000, 0.09, progress_messages=False)

# Forecasting process
prediction_of_test_data = MyLinearRegression().get_hypothesis(test_data, theta)

# score of model
print("Accuracy score of the Linear Regression: {} %".format(round(r2_score(test_labels, prediction_of_test_data)*100, 2)))
