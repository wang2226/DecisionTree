import pandas as pd
from sys import argv
from math import log, pow
import numpy as np
import copy


class decisionTree():
    # Create a dictionary to hold the tree.  This has to be outside the function so we can access it later.

    columns = ['WORKCLASS', 'EDUCATION', 'MARITAL_STATUS', 'OCCUPATION',
               'RELATIONSHIP', 'RACE', 'SEX', 'NATIVE_COUNTRY']

    # Constructor
    def __init__(self, trainingFile, testFile, model):
        self.trainingFile = trainingFile
        self.testFile = testFile
        self.model = model

    # Load data set
    def load(self, file_name):
        names = ['WORKCLASS', 'EDUCATION', 'MARITAL_STATUS', 'OCCUPATION',
                 'RELATIONSHIP', 'RACE', 'SEX', 'NATIVE_COUNTRY', 'SALARYlEVEL']

        X = pd.read_csv(file_name, sep=',', quotechar='"', header=0, engine='python')
        X.columns = names
        data = X.as_matrix()
        return data

    # Print a decision tree
    def print_tree(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[%s == %s]' % ((depth * ' ', (node['Node']), node['Value'])))
            self.print_tree(node['Left'], depth + 1)
            self.print_tree(node['Right'], depth + 1)
        else:
            print('%s[%s]' % ((depth * ' ', node)))

    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0

        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # Calculate entropy
    def calcEntropy(self, dataset):
        # Compute the counts of each unique value in the column.
        num_entries = len(dataset)
        label_counts = {}

        for feat_vec in dataset:  # the the number of unique elements and their occurance
            current_label = feat_vec[-1]
            if current_label not in label_counts.keys():
                label_counts[current_label] = 0

            label_counts[current_label] += 1

        # Initialize the entropy to 0.
        entropy = 0.0

        # Loop through the probabilities, and add each one to the total entropy.
        for key in label_counts:
            prob = float(label_counts[key]) / num_entries
            if prob > 0.0:
                entropy += prob * log(prob, 2)  # log base 2

        return -entropy

    # Split a dataset based on an attribute and an attribute value

    def test_split(self, index, value, dataset):

        left, right = list(), list()

        for row in dataset:
            if row[index] == value:
                left.append(row)
            else:
                right.append(row)
        return np.asarray(left), np.asarray(right)

    # Calculate information gain given a dataset, column to split on, and target.
    def calc_information_gain(self, data, index):
        # Calculate original entropy.
        original_entropy = self.calcEntropy(data)

        # Loop through the splits, and calculate the subset entropy.

        max_ig = 0.0
        ret_value = -1
        groups = ()

        for value in np.unique(data[:, index]):
            test_groups = self.test_split(index, value, data)
            to_subtract = 0.0

            for subset in test_groups:
                if subset.shape[0] > 0:
                    prob = (float(subset.shape[0]) / float(data.shape[0]))
                    to_subtract += prob * self.calcEntropy(subset)

            current_ig = original_entropy - to_subtract

            if current_ig > max_ig:
                max_ig = current_ig
                groups = test_groups
                ret_value = value

        # Return information gain.
        return max_ig, groups, ret_value

    # Select the best split point for a dataset
    def get_split(self, dataset, columns):
        b_node, b_value, b_score, b_groups = 'Nothing', 'Nothing', 0.0, None

        # Loop through and compute information gains.
        # for index in range(len(columns)):
        for index in range(len(columns)):
            gain, groups, value = self.calc_information_gain(dataset, index)
            if gain > b_score:
                b_node, b_value, b_score, b_groups = columns[index], value, gain, groups
        return {'Node': b_node, 'Value': b_value, 'Groups': b_groups}

    # Create a terminal node value
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def count_label(selfself, child):
        return [row[-1] for row in child]

    # Create child splits for a node or make terminal
    def split(self, mytree, columns, max_depth, depth):
        left, right = mytree['Groups']
        del (mytree['Groups'])
        sub_columns = columns[:]
        sub_columns.remove(mytree['Node'])
        index = columns.index(mytree['Node'])
        left = np.delete(left, index, axis=1)
        right = np.delete(right, index, axis=1)

        if not sub_columns:
            mytree['Left'], mytree['Right'] = self.to_terminal(left), self.to_terminal(right)
            return

        # check for a no split
        if not left.tolist() or not right.tolist():
            mytree['Left'] = mytree['Right'] = self.to_terminal(left + right)
            return

        # check for max depth
        if depth >= max_depth:
            mytree['Left'], mytree['Right'] = self.to_terminal(left), self.to_terminal(right)
            return

        # process left child
        label_list = self.count_label(left)

        if label_list.count(label_list[0]) == len(label_list):
            mytree['Left'] = self.to_terminal(left)
        else:
            mytree['Left'] = self.get_split(left, sub_columns)
            # if information gain is zero
            if mytree['Left']['Node'] == 'Nothing':
                mytree['Left'] = self.to_terminal(left)
            else:
                self.split(mytree['Left'], sub_columns, max_depth, depth + 1)

        # process right child
        label_list = self.count_label(right)

        if label_list.count(label_list[0]) == len(label_list):
            mytree['Right'] = self.to_terminal(right)
        else:
            mytree['Right'] = self.get_split(right, sub_columns)
            # if information gain is zero
            if mytree['Right']['Node'] == 'Nothing':
                mytree['Right'] = self.to_terminal(right)
            else:
                self.split(mytree['Right'], sub_columns, max_depth, depth + 1)

    # Build a decision tree
    def build_tree(self, train, max_depth):
        root = self.get_split(train, decisionTree.columns)
        self.split(root, decisionTree.columns, max_depth, 1)
        return root

    # Make a prediction with a decision tree
    def predict(self, mytree, row):
        columns = ['WORKCLASS', 'EDUCATION', 'MARITAL_STATUS', 'OCCUPATION',
                   'RELATIONSHIP', 'RACE', 'SEX', 'NATIVE_COUNTRY']
        if row[columns.index(mytree['Node'])] == mytree['Value']:
            if isinstance(mytree['Left'], dict):
                return self.predict(mytree['Left'], row)
            else:
                return mytree['Left']
        else:
            if isinstance(mytree['Right'], dict):
                return self.predict(mytree['Right'], row)
            else:
                return mytree['Right']

    def is_tree(self, obj):
        return (type(obj).__name__ == 'dict')

    def testing_major(self, major, data_test):
        error = 0.0
        for i in range(len(data_test)):
            if major != data_test[i]:
                error += 1
                # print 'major %d' %error
        return float(error)

    def prune(self, tree, test_data):
        # if have no test data collapse the tree
        if test_data.shape[0] == 0:
            return '>50K'

        left_set = []
        right_set = []
        # if the branches are not trees try to prune them
        if (self.is_tree(tree['Right']) or self.is_tree(tree['Left'])):
            left_set, right_set = self.test_split(decisionTree.columns.index(tree['Node']), tree['Value'], test_data)

        if self.is_tree(tree['Left']):
            tree['Left'] = self.prune(tree['Left'], left_set)

        if self.is_tree(tree['Right']):
            tree['Right'] = self.prune(tree['Right'], right_set)

        # if they are now both leafs, see if can merge them
        if not self.is_tree(tree['Left']) and not self.is_tree(tree['Right']):
            left_set, right_set = self.test_split(decisionTree.columns.index(tree['Node']), tree['Value'], test_data)

            if left_set.shape[0] == 0:
                left_error_sum = 0
            else:
                left_error_sum = self.testing_major(tree['Left'], left_set[:, -1])

            if right_set.shape[0] == 0:
                right_error_sum = 0
            else:
                right_error_sum = self.testing_major(tree['Right'], right_set[:, -1])

            error_no_merge = pow(left_error_sum, 2) + pow(right_error_sum, 2)
            tree_mean = self.to_terminal(test_data)
            error_merge = pow(self.testing_major(tree_mean, test_data[:, -1]), 2)

            if error_merge < error_no_merge:
                # print "merging"
                return tree_mean
            else:
                return tree
        else:
            return tree

class vanillaTree(decisionTree):
    def __init__(self, trainingFile, testFile, model, trainingPercent):
        decisionTree.__init__(self, trainingFile, testFile, model)
        self.trainingPercent = trainingPercent


class depthTree(decisionTree):
    def __init__(self, trainingFile, testFile, model, trainingPercent, validationPercent, maxDepth):
        decisionTree.__init__(self, trainingFile, testFile, model)
        self.trainingPercent = trainingPercent
        self.validationPercent = validationPercent
        self.maxDepth = maxDepth


class pruneTree(decisionTree):
    def __init__(self, trainingFile, testFile, model, trainingPercent, validationPercent):
        decisionTree.__init__(self, trainingFile, testFile, model)
        self.trainingPercent = trainingPercent
        self.validationPercent = validationPercent


# Create a dictionary to hold the tree.  This has to be outside the function so we can access it later.
tree = {}
# This list will let us number the nodes.  It has to be a list so we can access it inside the function.
nodes = []

if __name__ == '__main__':

    training_file = argv[1]
    test_file = argv[2]
    model = argv[3]
    training_percent = argv[4]

    # Implement a binary decision tree with no pruning using the ID3 algorithm
    if model == "vanilla":
        vani_tree = vanillaTree(training_file, test_file, model, training_percent)

        train = vani_tree.load(training_file)
        subtrain = copy.deepcopy(train)
        subtrain = train[0:int(len(train) * int(training_percent) / 100), :]

        max_depth = float("inf")

        tree = vani_tree.build_tree(subtrain, max_depth)

        # vani_tree.print_tree(tree)
        # print(tree)

        predictions_train = list()
        for row in subtrain:
            pd_train = vani_tree.predict(tree, row)
            predictions_train.append(pd_train)

        accuracy = vani_tree.accuracy_metric(subtrain[:, -1], predictions_train) / 100
        print("Training set accuracy: %.4f" % accuracy)

        test = vani_tree.load(test_file)
        predictions_test = list()
        for row in test:
            pd_test = vani_tree.predict(tree, row)
            predictions_test.append(pd_test)

        accuracy = vani_tree.accuracy_metric(test[:, -1], predictions_test) / 100
        print("Test set accuracy: %.4f" % accuracy)

    # Implement a binary decision tree with a given maximum depth
    elif model == "depth":
        validation_percent = argv[5]
        max_depth = int(argv[6])

        # Create depthTree object
        depth_tree = depthTree(training_file, test_file, model, training_percent, validation_percent, max_depth)

        # Read training data from file
        data_set = depth_tree.load(training_file)
        train_set = copy.deepcopy(data_set)
        validation_set = copy.deepcopy(data_set)

        # Prepare training data set
        train_set = data_set[0:int(len(data_set) * int(training_percent) / 100), :]

        # Build decision tree of max_depth
        tree = depth_tree.build_tree(train_set, max_depth)

        # Prepare validation data set
        validation_set = validation_set[int(len(validation_set) * (100-int(validation_percent)) / 100):, :]

        # Prepare test data set
        test_set = depth_tree.load(test_file)

        predictions_train = list()
        for row in train_set:
            pd_train = depth_tree.predict(tree, row)
            predictions_train.append(pd_train)

        accuracy = depth_tree.accuracy_metric(train_set[:, -1], predictions_train) / 100
        print("Training set accuracy: %.4f" % accuracy)

        predictions_validation = list()
        for row in validation_set:
            pd_validation = depth_tree.predict(tree, row)
            predictions_validation.append(pd_validation)

        accuracy = depth_tree.accuracy_metric(validation_set[:, -1], predictions_validation) / 100
        print("Validation set accuracy: %.4f" % accuracy)

        predictions_test = list()
        for row in test_set:
            pd_test = depth_tree.predict(tree, row)
            predictions_test.append(pd_test)

        accuracy = depth_tree.accuracy_metric(test_set[:, -1], predictions_test) / 100
        print("Test set accuracy: %.4f" % accuracy)

        # depth_tree.print_tree(tree)

    # Implement a binary decision tree with post-pruning using reduced error pruning
    elif model == "prune":
        validation_percent = argv[5]

        # Create pruneTree object
        prune_tree = pruneTree(training_file, test_file, model, training_percent, validation_percent)

        # Read training data from file
        data_set = prune_tree.load(training_file)
        train_set = copy.deepcopy(data_set)
        validation_set = copy.deepcopy(data_set)

        max_depth = float("inf")

        # Prepare training data set
        train_set = data_set[0:int(len(data_set) * int(training_percent) / 100), :]

        # Prepare validation data set
        validation_set = validation_set[int(len(validation_set) * (100 - int(validation_percent)) / 100):, :]

        # Build decision tree of max_depth
        tree = prune_tree.build_tree(train_set, max_depth)

        # Prepare test data set
        test_set = prune_tree.load(test_file)

        # Build decision tree with post-pruning using reduced error pruning
        post_prune_tree = prune_tree.prune(tree, validation_set)

        predictions_train = list()
        for row in train_set:
            pd_train = prune_tree.predict(post_prune_tree, row)
            predictions_train.append(pd_train)

        accuracy = prune_tree.accuracy_metric(train_set[:, -1], predictions_train) / 100
        print("Training set accuracy: %.4f" % accuracy)

        predictions_test = list()
        for row in test_set:
            pd_test = prune_tree.predict(post_prune_tree, row)
            predictions_test.append(pd_test)

        accuracy = prune_tree.accuracy_metric(test_set[:, -1], predictions_test) / 100
        print("Test set accuracy: %.4f" % accuracy)

        # prune_tree.print_tree(tree)

    else:
        print("Usage: python decisiontree.py ./path/to/file1.csv ./path/to/file2.csv " 
              "model trainingPercent, validationPercent, maxDepth")
