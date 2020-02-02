# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:51:43 2020

@author: shris
"""
#import numpy,pandas,sklearn ,xgboost lib
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# read the data into a DataFrame and get the first 5 records.
df=pd.read_csv('C:\\Users\\shris\\Desktop\\shristi\\deep learn\\pariksons diseease\\parkinsons.data')

 #extract features and labels from the dataset.
 #. The ‘status’ column has values 0 and 1 as labels

features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values

#Initialize a MinMaxScaler and scale the features to between -1 and 1 to normalize them. The MinMaxScaler transforms features by scaling them to a given range. The fit_transform() method fits to the data and then transforms it. We don’t need to scale the labels.
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels
#Now, split the dataset into training and testing sets keeping 20% of the data for testing.
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
#Initialize an XGBClassifier and train the model. .

model=XGBClassifier()
model.fit(x_train,y_train)

#Finally, generate y_pred (predicted values for x_test) and calculate the accuracy for the model. Print it out.
y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)