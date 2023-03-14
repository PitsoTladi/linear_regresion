# import modules
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn import linear_model


# create linear regression function
def linear_function(x_point, y_point):
    # find mean for the x and y coordinates
    x_mean = np.mean(x_point)
    y_mean = np.mean(y_point)

    # calculate the gradient between the points
    m = np.dot(np.transpose(x_point - x_mean), y_point - y_mean) / np.dot(np.transpose(x_point - x_mean),
                                                                          x_point - x_mean)
    b = y_mean - m * x_mean
    # return m and b
    return [m, b]


# Load the diabetes dataset
data = load_diabetes()
# Use only one feature
data_X = data.data[:, np.newaxis, 2]

# Split the data into training/testing sets
data_xtrain = data_X[:-20]
data_xtest = data_X[-20:]

# Split the targets into training/testing sets
data_ytrain = data.target[:-20]
data_ytest = data.target[-20:]

# predict the data
m, b = linear_function(data_xtrain, data_ytrain)
y_predict = b + m * data_X

# Plot outputs
plt.scatter(data_xtest, data_ytest, color='green', label='testing data')

plt.scatter(data_xtrain, data_ytrain, color='red', label='training data')

plt.plot(data_X, y_predict, c='b', linewidth=1)

plt.legend(loc='upper left')

# display scatter plot
plt.show()

'''
RESOURCES:
>https://www.stackoverflow.com/questions/55696942/how-to-apply-y-mx-b-formula-in-python-in-order-to-get-my-regression-line
>https://www.geekforgeeks.org>solving-linear-regression-in-python
>https://www.codegrapper.com/code-examples/python/y%3Dmx%2Bb+python
'''
