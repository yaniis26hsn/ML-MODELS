from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import numpy as np
import pandas as pd 
# this is a model about houses values in boston 
data = pd.read_csv("hou_all.csv",sep="," , header=None)

data = data.iloc[:, :-1] # removing that bias

X = data.iloc[:, :-1].values 
Y = data.iloc[:, -1].values

m = len(X)
split = int(0.75 * m)

X_test  = X[split:]
X_train = X[:split]
Y_test  = Y[split:]
Y_train = Y[:split]

myScaler = StandardScaler()

X_train = myScaler.fit_transform(X_train)
X_test = myScaler.transform(X_test)

model = SGDRegressor(
    max_iter= 5000 ,
    learning_rate= "constant" ,
    eta0= 0.005 ,
    penalty="l2",
    random_state= 42
)

model.fit(X_train , Y_train)

#predictions = model.predict(X_test) if you want to visualize the predictions

r2 = model.score(X_test , Y_test)

print(f"the r2 score is : {r2}")
