from decision_tree import build_tree, classify_tree
# Data ---> a list of lists where each sublist resembles an object containing attributes (values in the list) e.g data = [
#                                                                                                                           [0, 1, 3, 4, 6],
#                                                                                                                           [1, 4, 7, 9, 3]
#                                                                                                                                               ]
# Labels ---> a list of values resembling classes that each of the lists (objects) belongs to e.g labels = [1, 3]. Note! len(labels) = len(data)
# Test point ---> the same as with ther data list which is a list of lists where each sublist is an object with different attributes. The test point is an object with attributes
#
data = []
labels = []
test_point = []

# Build a tree using the data and labels
tree = build_tree(data, labels)

# Let the tree classify the new data point and see if it correctly predicts its class
print(classify_tree(test_point, tree))

