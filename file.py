import numpy as np
import pandas as pd
import DecisionTreeID3 as id3


def read_csv(name='data/data_train_test.csv'):
    df = pd.read_csv(name)
    col = df.columns
    X = df[col[:len(col)-1]].to_numpy()
    y = df[col[len(col)-1]].to_numpy()
    return X,y,col

X,y,col=read_csv(name='data/cay_phat_trien_moi.csv')

model = id3.DecisionTree()
model.build(X,y)

print(model.feature)
print(model.threshold)

print(model.left)
print(model.right)
print(model.value)

mt = np.array([model.left,model.right,model.value])
df =  pd.DataFrame(mt)
df.to_excel('data/quan_sat.xlsx',index=False)
# for val in x[:10]:   
#     print(model.predict(val))
# print(model.predict(x[4]))
