import numpy as np
import pandas as pd
import DecisionTreeID3 as id3

df = pd.read_csv("data/data_train_test.csv")

x=df[["Anh_sang","Tuoi_nuoc_deu","Do_am_dat","Chat_luong_dat","Phan_bon"]].to_numpy()
y=df["Phat_trien_tot"].to_numpy()


df = pd.read_csv("data/test.csv")
x_test=df[["Anh_sang","Tuoi_nuoc_deu","Do_am_dat","Chat_luong_dat","Phan_bon"]].to_numpy()


model = id3.DecisionTree()

model.build(x,y)

print(model.feature)
print(model.threshold)

print(model.left)
print(model.right)
print(model.value)

mt = np.array([model.left,model.right,model.value])
df =  pd.DataFrame(mt)
df.to_excel('data/quan_sat.xlsx',index=False)
for val in x[:10]:   
    print(model.predict(val))
# print(model.predict(x[4]))
