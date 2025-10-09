import pandas as pd
import matplotlib as plt
import numpy as np
import DecisionTreeID3 as ID3
from VeCay import VeCay


def read_csv(name='data/data_train_test.csv'):
    df = pd.read_csv(name)
    col = df.columns
    X = df[col[:len(col)-1]].to_numpy()
    y = df[col[len(col)-1]].to_numpy()
    return X,y,col

X,y,col=read_csv()

tree = ID3.DecisionTree()
tree.build(X, y)
print(tree.feature)
print(tree.threshold)
print(tree.left)
print(tree.right)
print(tree.value)

vc=VeCay.VeCay(tree,col)
vc.show_tree()

    
