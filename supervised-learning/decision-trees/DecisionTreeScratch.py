import Node
import math
from collections import Counter
import matplotlib.pyplot as plt

class DecisionTreeScratch:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    # --- Utilities ---
    def _entropy(self, y):
        counts = Counter(y)
        total = len(y)
        ent = 0.0
        for c in counts.values():
            p = c / total
            ent -= p * math.log2(p) if p > 0 else 0
        return ent

    def _info_gain(self, y, y_left, y_right):
        parent_entropy = self._entropy(y)
        n = len(y)
        n_l = len(y_left)
        n_r = len(y_right)
        if n_l == 0 or n_r == 0:
            return 0
        child_entropy = (n_l / n) * self._entropy(y_left) + (n_r / n) * self._entropy(y_right)
        return parent_entropy - child_entropy

    def _majority_class(self, y):
        return Counter(y).most_common(1)[0][0]
    
    
    def _best_split(self, X, y):
        best_gain = 0
        best_idx, best_thr = None, None
        n_features = len(X[0])
        # For each feature
        for feature_idx in range(n_features):
            # collect all unique values for that feature
            values = sorted(set(row[feature_idx] for row in X))
            if len(values) <= 1:
                continue
            # candidate thresholds: midpoints between consecutive unique values
            thresholds = [(values[i] + values[i+1]) / 2.0 for i in range(len(values)-1)]
            for thr in thresholds:
                left_y, right_y = [], []
                for xi, yi in zip(X, y):
                    if xi[feature_idx] <= thr:
                        left_y.append(yi)
                    else:
                        right_y.append(yi)
                gain = self._info_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = feature_idx
                    best_thr = thr
        return best_idx, best_thr, best_gain


    
    # Build tree recursively
    def _build_tree(self, X, y, depth=0):
        num_samples = len(y)
        num_labels = len(set(y))

        # stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or num_labels == 1 or num_samples < self.min_samples_split:
            leaf_value = self._majority_class(y)
            return Node(value=leaf_value)

        feat_idx, thr, gain = self._best_split(X, y)
        if feat_idx is None or gain == 0:
            return Node(value=self._majority_class(y))

        left_X, left_y, right_X, right_y = [], [], [], []
        for xi, yi in zip(X, y):
            if xi[feat_idx] <= thr:
                left_X.append(xi); left_y.append(yi)
            else:
                right_X.append(xi); right_y.append(yi)

        # In case a split created an empty partition (shouldn't happen if gain>0), make a leaf
        if len(left_y) == 0 or len(right_y) == 0:
            return Node(value=self._majority_class(y))

        left_node = self._build_tree(left_X, left_y, depth + 1)
        right_node = self._build_tree(right_X, right_y, depth + 1)
        return Node(feature_index=feat_idx, threshold=thr, left=left_node, right=right_node)

    def fit(self, X, y):
        """
        X: list of lists (n_samples x n_features)
        y: list of labels (n_samples)
        """
        self.root = self._build_tree(X, y)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        """
        X: list of lists (n_samples x n_features)
        returns list of predictions
        """
        return [self._predict_one(x, self.root) for x in X]

    # optional: print tree structure (simple)
    def _print_node(self, node, feature_names=None, depth=0):
        indent = "  " * depth
        if node.value is not None:
            print(f"{indent}Leaf: class={node.value}")
        else:
            fname = feature_names[node.feature_index] if feature_names else f"f{node.feature_index}"
            print(f"{indent}[{fname} <= {node.threshold:.4f}]")
            self._print_node(node.left, feature_names, depth+1)
            self._print_node(node.right, feature_names, depth+1)

    def print_tree(self, feature_names=None):
        self._print_node(self.root, feature_names, 0)

    def plot_custom_tree(node, feature_names, x=0.5, y=1.0, dx=0.2, dy=0.1, ax=None, depth=0):
    
      if ax is None:
          fig, ax = plt.subplots(figsize=(12, 6))
          ax.axis('off')
          plot_custom_tree(node, feature_names, x, y, dx, dy, ax, depth)
          plt.show()
          return

      # Draw current node
      if node.value is not None:
          text = f"Leaf\nClass={node.value}"
          box_color = "#AED581"
      else:
          feat = feature_names[node.feature_index]
          text = f"{feat} ≤ {node.threshold:.2f}"
          box_color = "#64B5F6"

      ax.text(x, y, text, ha='center', va='center',
              bbox=dict(boxstyle="round,pad=0.4", fc=box_color, ec='black', lw=1.5))

      # Draw children recursively
      if node.left:
          x_left = x - dx / (depth + 1)
          y_left = y - dy
          ax.plot([x, x_left], [y - 0.02, y_left + 0.05], 'k-')
          plot_custom_tree(node.left, feature_names, x_left, y_left, dx, dy, ax, depth + 1)
      if node.right:
          x_right = x + dx / (depth + 1)
          y_right = y - dy
          ax.plot([x, x_right], [y - 0.02, y_right + 0.05], 'k-')
          plot_custom_tree(node.right, feature_names, x_right, y_right, dx, dy, ax, depth + 1)