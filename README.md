# SGD-Regressor-for-Multivariate-Linear-Regression
## Date : 30 : 1 : 26
## reg no : 25010763
## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load California housing data, select features and targets, and split into training and testing sets.
2.Scale both X (features) and Y (targets) using StandardScaler.
3.Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data.
4.Predict on test data, inverse transform the results, and calculate the mean squared error.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Avinash Karthick B M
RegisterNumber:  25010763
*/
from sklearn.linear_model import SGDRegressor
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1,2],[2,1],[3,4],[4,3],[5,5]])
y = np.array([5,6,9,10,13])

model = SGDRegressor(max_iter=1000, eta0=0.01, learning_rate='constant')

model.fit(X, y)

print("Weights:", model.coef_)
print("Bias:", model.intercept_)

y_pred = model.predict(X)

plt.scatter(y, y_pred)
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted (SGDRegressor)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--') 
plt.show()

```

## Output:
<img width="744" height="504" alt="Screenshot 2026-01-30 141009" src="https://github.com/user-attachments/assets/976fc222-5b02-471a-aed0-4f47a0db1a60" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
