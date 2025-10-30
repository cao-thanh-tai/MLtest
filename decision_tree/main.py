import pandas as pd
import numpy as np


from src.decision_tree import decision_tree
from src.VeCay import VeCay

X_train = pd.read_csv('data/train.csv')
X_test = pd.read_csv('data/test.csv')

S_train=X_train.drop(['Survived',"Name","Age","Cabin","PassengerId","Embarked","Ticket"], axis= 1)
col = S_train.columns
S_train=S_train.to_numpy()
print(col)
y_train=X_train['Survived'].to_numpy()

S_test=X_test.drop(["Name","Age","Cabin","PassengerId","Embarked","Ticket"], axis= 1).to_numpy()


model = decision_tree(max_depth=10)
model.build(S_train,y_train)
y_pred = []

# print(S_train[:5])
# print(S_test[:5])
root = model.root

def tree(node,i):
    if i > 5 : return
    if node:
        tree(node.left,i+1)
        tree(node.right,i+1)
    return
# tree(root,0)
# for s in S_train:
#     y_pred.append(model.predict(s))
# print(y_pred[:10])
# mask = y_pred == y_train
# print(np.sum(mask)/len(mask))

y_pred_test = []
for s in S_test:
    y_pred_test.append(int(model.predict(s)))

submission = pd.DataFrame({
    'PassengerId': X_test['PassengerId'],
    'Survived': y_pred_test
})
submission.to_csv('submission.csv',index=False)



