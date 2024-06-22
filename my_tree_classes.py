import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score


class Node:
    def __init__(self,
                 attribute=None,
                 children=None,
                 value=None  # class
                 ):
        self.attribute = attribute
        self.children = children

        self.value = value  # if leaf node

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.valid_acc = None

    # def validate(self, X_val, y_val):
    #     y_pred = self.predict(X_val)
    #     self.valid_acc = accuracy_score(y_pred, y_val)

    def fit(self, X, y):
        self.n_attributes = X.shape[1]
        self.tree = self.grow_tree(X, y, depth=0)
        # self.validate(X_val, y_val)

    def grow_tree(self, X, y, depth):
        num_samples, num_attributes = X.shape[0], X.shape[1]
        num_labels = len(np.unique(y))

        # Warunki stopu
        if (self.max_depth is not None and depth >= self.max_depth) or num_labels == 1:
            return Node(value=np.argmax(np.bincount(y)))

        best_attribute = self.best_attribute(X, y)

        children = {}
        for value in np.unique(X[:, best_attribute]):
            # wybieramy wiersze (próbki) do tworzenia drzewa dalej
            child_indices = X[:, best_attribute] == value
            child_tree = self.grow_tree(X[child_indices], y[child_indices], depth + 1)
            # przypisujemy w słowniku dziecko dla wcześniejszego węzła po atrybucie
            children[value] = child_tree

        return Node(best_attribute, children)


    def best_attribute(self, X, y):
        best_gain = -1
        best_attribute = None
        for attribute in range(self.n_attributes):
            gain = self.gain(X, y, attribute)
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute
        return best_attribute


    def gain(self, X, y, attribute):
        parent_entropy = self.entropy(y)
        child_entropy = 0
        for value in np.unique(X[:, attribute]):
            child_indices = X[:, attribute] == value
            child_entropy += np.sum(child_indices) / len(y) * self.entropy(y[child_indices])
        return parent_entropy - child_entropy

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def predict(self, X):
        return np.array([self.predict_instance(x, self.tree) for x in X])


    def predict_instance(self, x, tree):
        if tree.value is not None:
            return tree.value
        if x[tree.attribute] in tree.children:
            return self.predict_instance(x, tree.children[x[tree.attribute]])
        else:
        # Zliczenie liczby wystąpień klas w dzieciach węzła
            child_values = [child.value for child in tree.children.values() if child.value is not None]
            if child_values:
                most_common_value = Counter(child_values).most_common(1)[0][0]  # Najczęściej występująca klasa
                return most_common_value
            else:
                return None