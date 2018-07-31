#SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling #we need to do this since svr class doesn't do this on its own
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
# Create your regressor here
from sklearn.svm import SVR
regressor = SVR(kernel ='rbf') #kernels decide if we want to do  poly  svr, linear svr, etc #rbf means gaussian kernel
regressor.fit(X,y)
# Predicting a new result
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]]))) #we use [[]] double brackets so that it's not a vector but an array

#Do inverse transform to get original scale of salary
y_pred = sc_y.inverse_transform(y_pred)


# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


