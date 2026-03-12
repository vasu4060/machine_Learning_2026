import numpy as np

def entropy(y):

          y_unique = np.unique(y)

          entropy_value = 0

          for label in y_unique:
                  
                    p = np.mean(y == label)

                    entropy_value -= p * np.log2(p+1e-9)
                   

          return round(entropy_value, 2)


y= [0,1,0,1,0]

print(entropy(y))



def information_gain(parent,left_child,right_child):
        
          parent_entropy = entropy(parent)
          left_child_entropy = entropy(left_child)
          right_child_entropy = entropy(right_child) 
          weight_left = len(left_child)/len(parent)
          weight_right = len(right_child)/len(parent)
          gain = parent_entropy - (weight_left * left_child_entropy + weight_right * right_child_entropy)    
          return round(gain, 2)         



parent = [0,0,0,0,1,1,1,1]
left = [0,0,0,1]
right = [0,1,1,1]


print(information_gain(parent,left,right))

def best_split(X, y):
        
        best_gain = -1
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]         
        for features in range(n_features):
                        thresholds = np.unique(X[:, features])
                        for threshold in thresholds:
                              left_indices = np.where(X[:, features] <= threshold)[0]
                              right_indices = np.where(X[:, features] > threshold)[0]
                              if len(left_indices) == 0 or len(right_indices) == 0:
                                          continue
                              gain = information_gain(y, y[left_indices], y[right_indices])
                              if gain > best_gain:
                                          best_gain = gain
                                          best_feature = features
                                          best_threshold = threshold
        return best_feature, best_threshold, best_gain    
                
                



def split_dataset(X,y,feature,threshold):
          X_left = X[X[:, feature] <= threshold]
          X_right = X[X[:, feature] > threshold]
          y_left = y[X[:, feature] <= threshold]
          y_right = y[X[:, feature] > threshold]
          return X_left, X_right, y_left, y_right





def build_tree(X, y, depth = 0):
          if len(np.unique(y)) == 1:
                    return y[0]
          feature, threshold, gain = best_split(X, y)
          if gain == 0:
                    return y[0]
          X_left, X_right, y_left, y_right = split_dataset(X, y, feature, threshold)
          left_subtree = build_tree(X_left, y_left, depth+1)
          right_subtree = build_tree(X_right, y_right, depth+1)
          return (feature, threshold, left_subtree, right_subtree)






X = [[2],[3],[10],[19]]
y = [0,0,1,1]
X_array = np.array(X)
y_array = np.array(y)

tree = build_tree(X_array, y_array)       
print(tree)