import pandas as pd
import matplotlib.pyplot as plt

points = pd.read_csv('data.csv')

def loss_function(m, b, points):
    total_error = 0
    # loop through number of rows or number of data points
    for i in range(len(points)): 
        # extract the x-coordinate from the ith row
        x = points.iloc[i, 0]
        # extract the y-coordinate from the ith row
        y = points.iloc[i, 1]
        # apply mean squared error function
        total_error += (y - (m * x + b)) ** 2
    # return the mean of the total error
    return total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = float(len(points))
    # loop through number of columns/data points
    for i in range(len(points)):
        # extract x coordinate for every row
        x = points.iloc[i, 0]
        # extract y coordinate for every row
        y = points.iloc[i, 1]
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

# create subplots
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.set_title("Linear Regression Model")
ax2.set_title("Loss Function vs Time")

# set limits on x and y axis of loss function plot
ax2.set_xlim([0,epochs])
ax2.set_ylim([0,loss_function(m,b,points)])

# plot data points
ax1.scatter(points.iloc[:,0], points.iloc[:,1])

# draw a starting line for regression model and starting point for loss function
line1, = ax1.plot(range(20, 80), range(20, 80), color='r')
line2, = ax2.plot(0, 0, color='r')

# apply gradient_descent function for each iteration to find m and b values
# apply loss function to find change in error as time (epochs) increases
xlist = []
ylist = []

for i in range(epochs):
    m, b = gradient_descent(m, b, points, L)
    line1.set_ydata(m * range(20, 80) + b)

    xlist.append(i)
    ylist.append(loss_function(m, b, points))
    line2.set_xdata(xlist)
    line2.set_ydata(ylist)

# show plot
plt.tight_layout()
plt.show()


