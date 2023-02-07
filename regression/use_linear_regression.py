import matplotlib.pyplot as plt
from linear_regression import gradient_descent

x_points = []
y_points = []
learning_rate = 0.001
iterations = 1000

# Find the best b and m values that fit the data
b, m = gradient_descent(x_points, y_points, 0.0001, 1000)
y_predictions = [x*m + b for x in x_points ]

# Plot original points and their predictions according to the m and b values
plt.plot(x_points, y_points, 'o')
plt.plot(x_points, y_predictions, 'o')
plt.show()