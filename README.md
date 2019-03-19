# Boston Housing Price

In this project, I want to use different algorithm to make prediction on Bosotn Housing Price.<br>
<br>
The data set is available on sklearn but I obtained the data set in csv for the convenience.

#### About the data set
The classic data set has 12 features and 1 column of response variable. 2 of the features are category variables.<br>
<br>
Use boston_dataset.DESCR in sklearn, you will see description of the data set.<br>
The category variables are 'chas' and 'rad'. 'chas' itself is a binary variable already so we do not need any additional feature engineering work; for 'rad', we will use one-hot-encoding since the data set is small and there is no correlation among classifiction on such column

## Language Used
I want to do the learning in both Python and R. We will use sklearn in Python. For the models I go over will be done in both language, but I will do additional diagnostic on linear regression just on R.

## Basic Models
I have done 4 basic models:<br>
1 - Simply Linear Regression
2 - Linear Regression
3 - Decision Tree Regressor
4 - Random forest Regressor

<br>
<br>
We will RMSE to compare among models.<br>
<br>
After learning all those models in Python and R, we found that Random Forest Regressor has the lowest RMSE and we will use this model for making prediction.