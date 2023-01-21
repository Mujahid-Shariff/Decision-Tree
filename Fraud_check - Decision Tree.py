# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 00:28:27 2022

@author: Mujahid Shariff
"""

import pandas as pd
df = pd.read_csv("Fraud_check.csv") 
df
df.head()

# Get information of the dataset
df.info()
print('The shape of our data is:', df.shape)
df.isnull().any()
df.dtypes

"""Data Visvualization & EDA (Exploratory Data Analysis)"""

# let's scatter plot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(data = df, hue = 'Taxable.Income')

# Boxplot
df.boxplot(column="Taxable.Income", vert=False)

# Histogram
df["Taxable.Income"].hist()

# Bar graph
df.plot(kind="bar")

# Kernel density estimation (KDE)
df.plot(kind="kde")

#Label encoding for our categorical variables
from sklearn.preprocessing import LabelEncoder 
LE = LabelEncoder()
df['Undergrad'] = LE.fit_transform(df['Undergrad'])
df['Marital.Status'] = LE.fit_transform(df['Marital.Status'])
df['Urban'] = LE.fit_transform(df['Urban'])

df.head()

# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:,1:])
df_norm.tail(10)

# Split the variables, X and Y variables
X = df_norm.drop(['Taxable.Income'], axis=1) # Independent Variable by leaving our target variable
Y = df_norm['Taxable.Income']   # Target Variable

# Converting the Taxable income variable to bucketing. 
df_norm["income"]="<=30000"
df_norm.loc[df["Taxable.Income"]>=30000,"income"]="Good"
df_norm.loc[df["Taxable.Income"]<=30000,"income"]="Risky"

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

# MODEL FITTING (Building Decision Tree)
from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor(max_depth = 5) 
regressor.fit(X_train, Y_train)

print("Node counts:",regressor.tree_.node_count)
print("max depth:",regressor.tree_.max_depth)

Y_pred_train = regressor.predict(X_train)
Y_pred_test = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
Training_err = mean_squared_error(Y_train, Y_pred_train).round(2)
Test_err = mean_squared_error(Y_test, Y_pred_test).round(2)

print("Training_error: ", Training_err.round(2))
print("Test_error: ", Test_err.round(2))

from sklearn.ensemble import BaggingRegressor
bag = BaggingRegressor(base_estimator = DecisionTreeRegressor(max_depth = 10),
                       n_estimators = 500, max_samples = 0.6,
                       random_state = 10, max_features = 0.7)

bag.fit(X_train, Y_train)
Y_pred_train = bag.predict(X_train)
Y_pred_test = bag.predict(X_test)
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