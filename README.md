# Linear Regression in Matrix Form

This project is about implementation of Linear Regression in Matrix Form. The equation of the best line will be got using batch gradient descent.

### Data

Data is collected from [RIA](www.ria.com) site. They contains information about the cost of renting flats in Lviv. The data's shape is only 95 rows and 9 columns.


### Technology

* python (3.6.2)
* numpy (1.15.1)
* scikit-learn (0.19.2)
* scipy (1.1.0)



### Description 

***Linear Regression*** tries to find a relationship between dependent variable ![y](https://latex.codecogs.com/gif.latex?y) and a set of independent variables ![variables](https://latex.codecogs.com/gif.latex?x_1%2C%20x_2%2C%20...%20%2C%20x_n). 
This relationship can be used to predict other values.

The hypothesis of the linear model is calculated by the formula:

![hypothesis](https://latex.codecogs.com/gif.latex?h_%5Ctheta%28x%29%20%3D%20%5Ctheta%5E%7BT%7Dx%3D%20%5Ctheta_0%20&plus;%20%5Ctheta_1x_1%20&plus;%20...%20&plus;%20%5Ctheta_nx_n)

Here, ![theta](https://latex.codecogs.com/gif.latex?%5Ctheta%20%3D%20%28%20%5Ctheta_0%2C%20%5Ctheta_1%2C%20...%20%2C%20%5Ctheta_n%29) are parameters or weights of the model.

This hypothesis is need to be trained. It means, that we need to find such parameters that allow to the predicted value to be as close as possible to the actual value.  In other words, the distance between hypothesis ![hypothesis](https://latex.codecogs.com/gif.latex?h_%5Ctheta%28x%29) and ![y](https://latex.codecogs.com/gif.latex?y) must be minimized.

Cost function is used to estimate this distance. It is using the function of least squares. This formula looks like this:

![cost function](https://latex.codecogs.com/gif.latex?J%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7B2n%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28h_%5Ctheta%28x%5E%7B%28i%29%7D%29%20-%20y%5E%7B%28i%29%7D%29%5E%7B2%7D)

And batch gradient descent is used for the training process. There each iteration performs update of parameters. It is the formula of batch gradient descent:

![batch gradient descent](https://latex.codecogs.com/gif.latex?%5Ctheta_j%20%3D%20%5Ctheta_j%20-%20%5Calpha%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28h_%5Ctheta%28x%5E%7B%28i%29%7D%29%20-%20y%5E%7B%28i%29%7D%29x_j%5E%7B%28i%29%7D)

Where ![alpha](https://latex.codecogs.com/gif.latex?%5Calpha) is a learning rate.

Gradient descent simply is an algorithm that makes small steps along a function to find a local minimum. 



### Getting starded

* Download this project or copy repository.
* Open project in your IDE for PYTHON.
* Run main.py file. 
