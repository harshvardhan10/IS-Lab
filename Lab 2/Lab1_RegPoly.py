##Import the Python libraries

import numpy as nm
import matplotlib.pyplot as mpl
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


##Define dataset


 
df = pd.DataFrame({"Stages" : (["Stage_1", "Stage_2", "Stage_3", "Stage_4", "Stage_5", "Stage_6", "Stage_7","Stage_8","Stage_9", "Stage_10" ]),
           "Temperature": ([1,2,3,4,5,6,7,8,9,10]),
           "Pressure":[0.45,0.5,0.6,0.8,1.1,1.5,2,3,5,10]})


##Extracted the dependent(y) and independent variable(x) from the dataset.

x = df[['Temperature']]
y = df['Pressure']



##Building the Linear regression model

 
linear_regs= LinearRegression()
linear_regs.fit(x,y)

##Building the Polynomial regression model

 
polynomial_regs= PolynomialFeatures(degree= 2)
x_poly= polynomial_regs.fit_transform(x)


##Next step is to use another LinearRegression object ,
##namely linear_reg_2, to fit your x_poly vector to the linear model.

 
linear_reg_2 =LinearRegression()
linear_reg_2.fit(x_poly, y)


##Linear regression Plot

 
mpl.scatter(x,y,color="blue")
mpl.plot(x,linear_regs.predict(x), color="red")
mpl.title("Lab 1 (Linear Regression)")
mpl.xlabel("Temperature")
mpl.ylabel("Pressure")
mpl.show()

##Polynomial Regression Plot

 
mpl.scatter(x,y,color="blue")
mpl.plot(x, linear_reg_2.predict(polynomial_regs.fit_transform(x)), color="red")
mpl.title("Truth detection model(Polynomial Regression)")
mpl.xlabel("Temperature")
mpl.ylabel("Pressure")
mpl.show()


##Linear Regression model

 
lin_pred = linear_regs.predict([[6.5]])
print("Linear Regression model Prediction  : " , lin_pred)

##Polynomial Regression model

poly_pred = linear_reg_2.predict(polynomial_regs.fit_transform([[6.5]]))
print("Polynomial Regression model Prediction : ", poly_pred)
















