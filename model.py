import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

def loss_function(m, b, data):
    total_error = 0
    # loop through number of rows or number of data points
    for i in range(len(data)): 
        # extract the x-coordinate from the ith row
        x = data.iloc[i, 0]
        # extract the y-coordinate from the ith row
        y = data.iloc[i, 1]
        # apply mean squared error function
        total_error += (y - (m * x + b)) ** 2
    # return the mean of the total error
    return total_error / float(len(data))

def gradient_descent(m_now, b_now, data, L):
    m_gradient = 0
    b_gradient = 0
    n = float(len(data))
    # loop through number of columns/data points
    for i in range(len(data)):
        # extract x coordinate for every row
        x = data.iloc[i, 0]
        # extract y coordinate for every row
        y = data.iloc[i, 1]
        # compute the gradient of error function wrt m
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        # compute the gradient of error function wrt b
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    # update m 
    m = m_now - L * m_gradient
    # update b
    b = b_now - L * b_gradient
    return [m, b]

# define parameters
m = 0
b = 0
L = 0.00001
epochs = 100

# train model
for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)

# print results
print("m:", m, "b:", b)

# use model to predict for new data points
def predict(x):
    return m * x + b

predict_points = [10.4, 24.5, 46.1, 60.5]

results = []

for i in predict_points:
    results.append(predict(i))

print("predictions:", results) # displays prediction corresponding to each input

