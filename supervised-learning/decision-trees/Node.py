class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index  # index of the feature to split on
        self.threshold = threshold          # threshold value for the split
        self.left = left                    # left child node (<= threshold)
        self.right = right                  # right child node (> threshold)
        self.value = value                  # class label if leaf