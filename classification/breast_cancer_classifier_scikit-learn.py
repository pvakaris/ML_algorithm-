# Using a decision tree on the breats cancer dataset.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load the data
breast_cancer = load_breast_cancer()

# Split the data to training and test data
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.1, random_state=0)

# Now create a decision tree and train it with the training data
breast_cancer_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)

# Get the accuracy on test data
accuracy = breast_cancer_tree.score(X_test, y_test)
print(accuracy)

#   DRAWING (only available to datasets that have 2 attributes). If you want the diagram representing the data, comment the code above and uncomment the code below
#
# plot_step = 0.02
# breast_cancer = load_breast_cancer()
# X = breast_cancer.data [:, [1, 3]] # 1 and 3 are the features we will use here.
# y = breast_cancer.target

# # Now create a decision tree and fit it to the iris data:
# breast_cancer_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X, y)

#
# # Now plot the decision surface that we just learnt by using the decision tree to
# # classify every packground point.
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#                      np.arange(y_min, y_max, plot_step))

# Z = breast_cancer_tree.predict(np.c_[xx.ravel(), yy.ravel()]) # Here we use the tree
#                                                      # to predict the classification
#                                                      # of each background point.
# Z = Z.reshape(xx.shape)
# cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

# # Also plot the original data on the same axes
# plt.scatter(X[:, 0], X[:, 1], c=y.astype(float))#, cmap='autumn')

# # Label axes
# plt.xlabel( breast_cancer.feature_names[1], fontsize=10 )
# plt.ylabel( breast_cancer.feature_names[3], fontsize=10 )

# plt.show()
