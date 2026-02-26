# logistic reg model: 

from sklearn.datasets import load_iris 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle


import numpy as np 


data = load_iris() 

X = data.data
Y = data.target

X, Y = shuffle(X, Y, random_state=42) # since the data set was ordered we should shuffle so that this won't hurt the learning 

split = int(0.8 * len(X) )

X_test = X[split:]
X_train = X[:split]
Y_test = Y[split:]
Y_train = Y[:split] 

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SGDClassifier(
        loss="log_loss" ,
        eta0= 0.05 ,
        penalty= "l2" ,
        learning_rate= "constant" ,
        random_state= 42 ,
        max_iter=4000 

) 

model.fit(X_train ,Y_train)



accuracy = model.score(X_test , Y_test)

print(f"the accuracy of the model is {accuracy}")
