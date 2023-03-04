from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# Showing AND OR and XOR classification line
data = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = [0, 0, 0, 1]

plt.scatter([p[0] for p in data], [p[1] for p in data], labels)
classifier = Perceptron(max_iter=40)

classifier.fit(data, labels)
print(classifier.score(data, labels))

x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)

point_grid = list(product(x_values, y_values))

distances = classifier.decision_function(point_grid)
abs_distances = [ abs(d) for d in distances ]
distances_matrix = np.reshape(abs_distances, (100, 100))

plt.pcolormesh(x_values, y_values, distances_matrix)

plt.show()
