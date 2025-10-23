import pandas as pd



from src.decision_tree import decision_tree

X_train = pd.read_csv('data/train.csv')

S_train=X_train.drop('Phat_trien_tot', axis= 1)
y_train=X_train['Phat_trien_tot']

print(X_train.columns)

model = decision_tree(max_depth=10)
model.build(S_train,y_train)


