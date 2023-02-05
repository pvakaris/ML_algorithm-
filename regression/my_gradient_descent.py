import numpy as np
from sklearn.datasets import make_regression
import math

def gradient_descent(x, y, learning_rate=0.01, tolerance=0.0001, max_iterations=10000):
    k = b = prev_cost = 0
    n = len(x)

    for i in range(max_iterations):
        # Find the predicted value of y according to the formula y = kx + b
        y_pred = k*x + b
        cost = (1/n) * sum([val**2 for val in (y-y_pred)])
        if(i > 0 and (math.isclose(cost, prev_cost, abs_tol=tolerance) or cost > prev_cost)):
            return (k, b)
        prev_cost = cost
        
        # Calculate the derivatives of k and b
        k_derivative = -(2/n) * sum(x * (y-y_pred))
        b_derivative = -(2/n) * sum(y-y_pred)
        
        # Update the values of k and b (descend)
        k = k - learning_rate * k_derivative
        b = b - learning_rate * b_derivative
        
        print('k = {}   | b = {}   | cost = {}   | iteration = {}'.format(k, b, cost, i))
        
    return (k, b)
        
        

X, y = make_regression(n_samples=20, n_features=1, noise=2, random_state=2345)

print(X)
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([5, 7, 9, 11, 13])
# print(gradient_descent(x, y))