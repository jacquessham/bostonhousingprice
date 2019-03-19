import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the data set
boston = pd.read_csv("boston.csv")
# Do one-hot-encoding on the column 'rad'
rad_dummies = pd.get_dummies(boston['rad'], prefix="rad")
boston = pd.concat([boston, rad_dummies], axis=1).drop(columns=['rad'])

# Split the data
X = boston.drop(['medv'], axis=1)
y = boston['medv']

# Split the data set to 80:20 split of training and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)

# First Model
X_train_slr = pd.DataFrame(X_train['rm'])
X_test_slr = pd.DataFrame(X_test['rm'])
slr = LinearRegression()
slr.fit(X_train_slr,y_train)
slr_score = slr.score(X_test_slr,y_test)
print(f"The First Model has R squared of {slr_score:.2f}")
slr_pred = slr.predict(X_test_slr) # Obtain the fitted value
slr_rmse = sqrt(mean_squared_error(y_test, slr_pred)) # Take the square root of MSE
print(f"The First Model has RMSE of {slr_rmse:.2f}")

# Second Model
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_score = lr.score(X_test, y_test)
print(f"The Second Model has R squared of {lr_score:.2f}")
lr_pred = lr.predict(X_test)
lr_rmse = sqrt(mean_squared_error(y_test, lr_pred))
print(f"The Second Model has RMSE of {lr_rmse:.2f}")

# Third Model
dt = DecisionTreeRegressor(random_state=49)
dt.fit(X_train,y_train)
dt_score = dt.score(X_test, y_test)
print(f"The Third Model has R squared of {dt_score:.2f}")
dt_pred = dt.predict(X_test)
dt_rmse = sqrt(mean_squared_error(y_test, dt_pred))
print(f"The Third Model has RMSE of {dt_rmse:.2f}")

# Forth Model
rf = RandomForestRegressor(random_state=49)
rf.fit(X_train,y_train)
rf_score = rf.score(X_test, y_test)
print(f"The Forth Model has R squared of {rf_score:.2f}")
rf_pred = rf.predict(X_test)
rf_rmse = sqrt(mean_squared_error(y_test, rf_pred))
print(f"The Forth Model has RMSE of {rf_rmse:.2f}")