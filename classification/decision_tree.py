from collections import Counter
import operator

# Author - Vakaris Paulavicius

# Leaf of the tree
class Leaf:
    def __init__(self, labels, value):
        self.labels = Counter(labels)
        self.value = value

# Non-leaf element of the tree
class Internal_Node:
    def __init__(self, feature, branches, value):
        self.feature = feature
        self.branches = branches
        self.value = value


# Splits the data set according to the column number (attribute number)
def split(dataset, labels, column):
    print(column)
    data_subsets = []
    label_subsets = []
    counts = list(set([data[column] for data in dataset]))
    counts.sort()
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(dataset)):
            if dataset[i][column] == k:
                new_data_subset.append(dataset[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    return data_subsets, label_subsets


# Calculate the GINI impurity of a given set of labels
def gini_impurity(labels):
    impurity = 1
    counts = Counter(labels)
    for count in counts:
        probability = counts[count] / len(labels)
        impurity -= probability ** 2
    
    return impurity

# The simple information gain
def information_gain(all_labels, split_labels):
    info_gain = gini_impurity(all_labels)
    for label_sublist in split_labels:
        info_gain -= gini_impurity(label_sublist)
        
    return info_gain

# The weighted information gain
def weighted_information_gain(all_labels, split_labels):
    info_gain = gini_impurity(all_labels)
    for label_sublist in split_labels:
        info_gain -= gini_impurity(label_sublist) * (len(label_sublist) / len(all_labels))
        
    return info_gain

# Find the feature that gives the biggest information gain when split upon as well as its gain
def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    for i in range(len(dataset[0])):
        split_datasets, split_labels = split(dataset, labels, i)
        gain = weighted_information_gain(labels, split_labels)
        if gain > best_gain:
            best_gain = gain
            best_feature = i
        
    return best_feature, best_gain

# Build a decision tree
def build_tree(dataset, labels, value = ""):
    best_feature, best_gain = find_best_split(dataset, labels)
    if best_gain == 0:
        return Leaf(Counter(labels), value)
    
    data_subsets, label_subsets = split(dataset, labels, best_feature)
    branches = []
    for i in range(len(data_subsets)):
        branch = build_tree(data_subsets[i], label_subsets[i], data_subsets[i][0][best_feature])
        branches.append(branch)
    return Internal_Node(best_feature, branches, value)

# Classify a new datapoint given the tree and the datapoint
def classify_tree(datapoint, tree):
    if isinstance(tree, Leaf):
        return max(tree.labels.items(), key=operator.itemgetter(1))[0]
    value = datapoint[tree.feature]
    for branch in tree.branches:
        if branch.value == value:
            return classify_tree(datapoint, branch)
        
# Used to evaluate how accurate a tree is (USED WITH TEST DATA)  
def get_tree_accuracy(tree, test_data, test_labels):
    correct_predictions = 0
    for i in range(len(test_data)):
        prediction = classify_tree (test_data[i], tree)
        if prediction == test_labels[i]:
            correct_predictions += 1
            
    return correct_predictions / len(test_labels)
