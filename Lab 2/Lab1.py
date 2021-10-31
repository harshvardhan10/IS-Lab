import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Import the Height Weight Dataset 
data = pd.read_csv('Poly_Dataset.csv') 

#Store the data in the form of dependent and independent variables separately
X = data.iloc[:, 0:1].values 
y = data.iloc[:, 1].values

from sklearn.model_selection import train_test_split

## Splitting dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


from sklearn.linear_model import LinearRegression

## Linear regression
LinReg = LinearRegression()
LinReg.fit(X_train, y_train)

##Plotting Linear regression
plt.scatter(X_train, y_train, color = 'green') 
plt.plot(X_train, LinReg.predict(X_train), color = 'blue')
plt.title('Linear Regression') 
plt.xlabel('Age') 
plt.ylabel('Height') 
  
plt.show()

##Polynomial regression
from sklearn.preprocessing import PolynomialFeatures 
  
polynom = PolynomialFeatures(degree = 2) 
X_polynom = polynom.fit_transform(X_train) 
  
##X_polynom

PolyReg = LinearRegression() 
PolyReg.fit(X_polynom, y_train)

plt.scatter(X_train, y_train, color = 'green') 
  
plt.plot(X_train, PolyReg.predict(polynom.fit_transform(X_train)), color = 'blue') 
plt.title('Polynomial Regression') 
plt.xlabel('Age') 
plt.ylabel('Height') 
  
plt.show()

y_predict_slr = LinReg.predict(X_test)

#Model Evaluation using R-Square for Simple Linear Regression
from sklearn import metrics
r_square = metrics.r2_score(y_test, y_predict_slr)
print('R-Square Error associated with Simple Linear Regression:', r_square)

y_predict_pr = PolyReg.predict(polynom.fit_transform(X_test))

#Model Evaluation using R-Square for Polynomial Regression
from sklearn import metrics
r_square = metrics.r2_score(y_test, y_predict_pr)
print('R-Square Error associated with Polynomial Regression is:', r_square)

PolyReg.predict(polynom.fit_transform([[53]]))
