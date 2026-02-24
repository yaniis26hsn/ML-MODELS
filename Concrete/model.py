import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

data = pd.read_excel("Concrete_Data.xls") 


X = data.iloc[: , :-1] 
Y = data.iloc[: , -1]

m = len(X) 
split = int(m*0.8)

X_test = X[split:]
X_train = X[:split]
Y_test = Y[split:]
Y_train= Y[:split]


Scaler = StandardScaler() 
y_scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)
Y_train_scaled = y_scaler.fit_transform(Y_train.reshape(-1, 1))
Y_test_scaled = y_scaler.transform(Y_test.reshape(-1 , 1)) 
# reshaping is necessary cz the fnc expects 2D array , not a vector  

model = SGDRegressor(
    eta0= 0.01 ,
    max_iter= 5000 ,
    learning_rate= "constant" ,
    penalty="l2" ,
    random_state= 42 
)

model.fit(X_train , Y_train_scaled.ravel())
#it must getback to 1D again that why i added ravel()

myPredictions = model.predict(X_test) # X_test is already scaled
myPredictions = y_scaler.inverse_transform(myPredictions.reshape(-1 , 1))

r2 = model.score(X_test , Y_test_scaled.ravel())

print(f"the r2 score is {r2}")