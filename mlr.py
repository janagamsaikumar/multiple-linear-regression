import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
dataset=pd.read_csv(r'C:\Users\saikumar\Desktop\AMXWAM data science\class 16 & 17_oct 17, 2020\2.MULTIPLE LINEAR REGRESSION\50_Startups.csv')
# this data set contains the company stats of different departments and have
# to predict that which department would give more returns instead 
# of investing in all the departments
#------------------------------------------------------------------------------
#find independent and dependent varibles in the dataset and seperate them
X=dataset.iloc[:,:4]
y=dataset.iloc[:,-1]
X.isnull().any() # this checks the dataset having null values or not if it 
# returns true there are no null values which are empty  have to treat using 
# univariate measures
y.isnull().any() # returned false dataset has null values
X=pd.get_dummies(X)
# the categorical data has turned to numerical dummy variable equaion would be m-1
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# here feature scaling is not required for MLR
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)# we have fit our reg model to training set and now have to test with y test
y_pred=reg.predict(X_test)
# but here we have multiple independent variables we cannot say our model is accuarte so we have to elimnicate the  variables which are less corelated with the dependent variables
import statsmodels.api as sm
import statsmodels.formula.api as smf
#sm is the statsmodel with the help of this we can calculate the caculations very easily without complexity
# we have y=c+m1x1....mnxn c is the constant we dont have constant value in the data set so we are going to add one constant valuen to all the variables
# we have added the one column to independent variable axis should be specified otherwise the values can be of any shape if mentioned values have a constant shape 
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
# now we have to start finding the varible which is high corelated with our dependent variables based on the pvalue
# when p values is < 0.05 it is high corelated if > 0.05 less corelated we can drop of that variables to make our model perfect
# and also observe the r-square and adjusted r-square 
# r-square is always > adjusted r-square
# if less than that ur model is not accurate to predict with that variable
X_opt=X[:,[0,1,2,3,4,5]]
 # we have created a new matrix by dropping of the last column of independent variables
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# endog would always be dependent variables
#exog would be independent 
X_opt=X[:,[0,1,2,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# now it is clear that investing in r&d will  give profits instead of putting in all the dept.
regressor_OLS.summary()






