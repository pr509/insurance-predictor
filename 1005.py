import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import pandas as pd
insurance=pd.read_csv('insurance.csv')
insurance_numberic=insurance.select_dtypes(['int64','float64'])
x=insurance_numberic.drop(['charges'],axis=1)
y=insurance_numberic['charges']
insurance_categorical=insurance.select_dtypes(include=['object'])
#print(insurance_categorical.head())
insurance_dummies=pd.get_dummies(insurance)
#print(insurance_dummies.head(3))
x=pd.concat([x,insurance_dummies],axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
model_pred=model.predict(x_test)
graph=pd.DataFrame({'actual':y_test,'predicted':model_pred})
print(graph)
from sklearn.metrics import r2_score
print(r2_score(y_test,model_pred))
