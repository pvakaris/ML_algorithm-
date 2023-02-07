from decision_tree import weighted_information_gain, split, classify_tree, Leaf, Internal_Node
import random
import math
import numpy as np
from collections import Counter

# Author - Vakaris Paulavicius 

# Give random a seed to make the results reproducable. SEED can be set to any value.
SEED = 123523
random.seed(SEED)
np.random.seed(SEED)

# This function returns a set of features of size n from the list of features provided as a parameter
def get_n_random_features(features):
    feature_amount = len(features)
    # The rule of thumb is that for trees in the random forests
    # the number of features used for each tree is square root of the number of all features
    n = round(math.sqrt(feature_amount))
    features = np.random.choice(feature_amount, n, replace = False)
    return features

# Used to run the test data on the forest and see ow accurate it is
def get_forest_accuracy(forest, test_data, test_labels):
    correct_predictions = 0
    for i in range(len(test_data)):
        predicted_labels = []
        for tree in forest:
            predicted_labels.append(classify_tree(test_data[i], tree))
        # Find the most popular label
        forest_prediction = max(predicted_labels,key=predicted_labels.count)
        if forest_prediction == test_labels[i]:
            correct_predictions += 1
            
    return correct_predictions / len(test_labels)

# Used to classify an example
def classify_forest(attributes, forest):
    predictions = []
    for tree in forest:
        predictions.append(classify_tree(attributes, tree))
    return max(predictions,key=predictions.count)
    
# Used to find best splits from limited amounts of features
def find_best_split_subset(dataset, labels):
    best_gain = 0
    best_feature = 0
    
    features = get_n_random_features(dataset[0])
    
    for feature in features:
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = weighted_information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain

# Build a decision tree
def build_tree_for_forest(dataset, labels, value = ""):
    best_feature, best_gain = find_best_split_subset(dataset, labels)
    if best_gain == 0:
        return Leaf(Counter(labels), value)
    data_subsets, label_subsets = split(dataset, labels, best_feature)
    branches = []
    for i in range(len(data_subsets)):
        branch = build_tree_for_forest(data_subsets[i], label_subsets[i], data_subsets[i][0][best_feature])
        branches.append(branch)
    return Internal_Node(best_feature, branches, value)

# Used to generate a list of trees of a particular size and train it using the traning data
def generate_forest(size, train_data, train_labels):
    trees = []
    for i in range(size):
        indices = [random.randint(0, len(train_data)-1) for x in range(len(train_data))]

        training_data_subset = [train_data[index] for index in indices]
        training_labels_subset = [train_labels[index] for index in indices]

        tree = build_tree_for_forest(training_data_subset, training_labels_subset)
        trees.append(tree)
        
    return trees
