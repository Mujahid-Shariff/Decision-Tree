# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 23:29:59 2022

@author: Mujahid Shariff
"""
import pandas as pd
df=pd.read_csv("Company_Data.csv")
df
df.head()
df.dtypes

# Get information on the dataset
df.info()
print('The shape of our data is:', df.shape)
df.isnull().any() #to find out if there are any null values
df.dtypes

"""Data Visvualization & EDA (Exploratory Data Analysis)"""

# let's scatterplot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(data = df, hue = 'Sales')

# Boxplot
df.boxplot(column="Sales", vert=False)

# Histogram
df["Sales"].hist()

# Bar graph
df.plot(kind="bar")

# Kernel density estimation (KDE)
df.plot(kind="kde")

#label encoding for our categorical variables
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['ShelveLoc']=LE.fit_transform(df['ShelveLoc'])
df['Urban']=LE.fit_transform(df['Urban'])
df['US']=LE.fit_transform(df['US'])

df.head()

# Mapping
df['ShelveLoc'] = df['ShelveLoc'].map({'Good':1,'Medium':2,'Bad':3})
df.head()

#splitting X and Y variables
X=df.iloc[:,1:11] # Independent Variable
Y=df['Sales'] # Target Variable

# Counts in Target variable
df.Sales.value_counts() #counting he number of data in our target variable
col = list(df.columns) #listing all the columns in our dataset
col

#Data Partition, splitting the data into Test and Training data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

#model fitting using Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor(max_depth = 5) 
regressor.fit(X_train, Y_train)
regressor

#calculating Node counts and max depth
print("Node Counts :",regressor.tree_.node_count)
print("Max Depth:",regressor.tree_.max_depth)

#Finding out Y Predicted values for Test and Train Data
Y_pred_train = regressor.predict(X_train)
Y_pred_test = regressor.predict(X_test)

#step 6 - Metrics
from sklearn.metrics import mean_squared_error
Training_err = mean_squared_error(Y_train, Y_pred_train).round(2)
Test_err = mean_squared_error(Y_test, Y_pred_test).round(2)

print("Training_error: ", Training_err.round(2))
print("Test_error: ", Test_err.round(2))

#Bagging Regressor
from sklearn.ensemble import BaggingRegressor
bag=BaggingRegressor(base_estimator= DecisionTreeRegressor(max_depth = 10),
                     n_estimators=500,max_samples=0.6,
                     random_state=10,max_features = 0.7) #here 500 is, model is building 500 trees to check errors

bag.fit(X_train,Y_train)
Y_pred_train= bag.predict(X_train)
Y_pred_test=bag.predict(X_test)
Training_err = mean_squared_error(Y_train, Y_pred_train).round(2)
Test_err = mean_squared_error(Y_test, Y_pred_test).round(2)

print("Training_error: ", Training_err.round(2))
print("Test_error: ", Test_err.round(2))

from sklearn import tree
tree.plot_tree(regressor)
import matplotlib.pyplot as plt
fn = ['Sales','CompPrice','Income','Advertising','Population','Price'] # features name = fn
cn = ['1', '2', '3'] # class name = cn
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (2,2), dpi=300)
tree.plot_tree(regressor, feature_names = fn, class_names = cn, filled = True)

