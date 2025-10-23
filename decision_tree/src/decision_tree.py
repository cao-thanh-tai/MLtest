import numpy as np

from .node import node

class decision_tree:
    def __init__(self, max_depth = None):
        self.root = None
        self.max_depth = max_depth
        
    def split_tree(self, S, y, threshold, feature_idx):
        if is_categorical(S[feature_idx]):
            mask = S[:, feature_idx] == threshold
        else:
            mask = S[:, feature_idx] <= threshold
        return S[mask], y[mask], S[~mask], y[~mask]
    
    def build(self, S, y):
        S = np.array(S)
        y = np.array(y)
        self.root = self.build_tree(S, y, 0)
        
    def build_tree(self, S, y, depth):
        if (is_pure(y) or depth == self.max_depth):
            return node(value = most_common_label(y))
        
        feature_idx, threshold = find_best_split(S, y)
        
        if feature_idx is None:
            return node(value=most_common_label(y))
        S_left, y_left, S_right, y_right = self.split_tree(S, y, threshold, feature_idx)
        left_node = self.build_tree(S_left, y_left, depth + 1)
        right_node = self.build_tree(S_right, y_right, depth + 1)
        
        return node(feature_idx, threshold, left_node, right_node)
    
    def predict(self, s):
        node = self.root
        while node and node.value is None:
            if isinstance(s[node.feature], (int, float)):
                node = node.left if s[node.feature] <= node.threshold else node.right
            else:
                node = node.left if s[node.feature] == node.threshold else node.right
        return node.value if node else "ko bt"
                
                
                
        
def entropy(y):
    if len(y) == 0 : return 0.0
    classes, count = np.unique(y, return_counts=True)
    probs = count/len(y)
    return -np.sum(probs*np.log(probs + 1e-10))

def coditional_entropy_categorical(s, y):
    """tinh entropy cho du lieu roi rac"""
    unique_values = np.unique(s)
    weightesd_entropy = 0
    for val in unique_values:
        mask = s == val
        weightesd_entropy += (np.sum(mask)/len(s))*entropy(y[mask])
    return weightesd_entropy, unique_values    

def coditional_entropy_continuous(s, y):
    """tinh entropy cho du lieu lien tuc"""
    sorted_indices = np.argsort(s)
    s_sorted, y_sorted = s[sorted_indices], y[sorted_indices]
    min_entropy = float('inf')
    best_threshold = None
    n = len(s)
    for i in range(n-1):
        if y_sorted[i] != y_sorted[i+1]:
            threshold = (s_sorted[i] + s_sorted[i+1])/2
            mask = s <= threshold
            entropy_temp = (np.sum(mask)/n)*entropy(y_sorted[mask]) + (np.sum(~mask)/n)*entropy(y_sorted[~mask])
            if min_entropy > entropy_temp:
                min_entropy=entropy_temp
                best_threshold = threshold
    return min_entropy, [best_threshold]
    
def information_gain(s, y):
    if is_categorical(s):
        coud_entropy, thresholds = coditional_entropy_categorical(s, y)
    else :
        coud_entropy, thresholds = coditional_entropy_continuous(s, y)
    return entropy(y) - coud_entropy, thresholds

def is_categorical(s):
    unique_values = np.unique(s)
    if (isinstance(s[0], (int, float))):
        return isinstance(s[0], int) and len(unique_values) < 10
    return True

def find_best_split(S, y):
    best_gain = -1
    best_threshold = None
    best_feature = None
    
    for i in range(len(S[0])):
        gain, thresholds = information_gain(S[:,i], y)
        if gain > best_gain:
            best_gain = gain
            best_threshold = thresholds[0] if thresholds[0] else None
            best_feature = i
    return best_feature, best_threshold

def is_pure(y):
    return len(y) == 0 or len(np.unique(y)) == 1

def most_common_label(y):
    if len(y) == 0 :
        return "ko bt"
    if np.issubdtype(y.dtype, np.integer):
        return np.argmax(np.bincount(y))
    uniques, count = np.unique(y, return_counts=True)
    return uniques[np.argmax(count)]
