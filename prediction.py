import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.preprocessing import MinMaxScaler

#Importing data-set
training_set = pd.read_csv('Downloads/AAPL.csv')

training_set = training_set.iloc[:,1:2].values
# s = MinMaxScaler()
# training_set = s.fit_transform(training_set)

X_train = training_set[0:127]
Y_train = training_set[1:128]

X_train = np.reshape(X_train, (127,1))
print(X_train.shape)
print("Output: \n", Y_train.shape)

model = LinearRegression()

model.fit(X_train, Y_train)
Y_pred = model.predict(X_train)
print('predict response:', Y_train,' Y_pred \n', Y_pred, sep='\n')

R_sq = model.score(X_train, Y_train)
print('Accuracy (coef of determination) = ', R_sq)

plt.scatter(X_train, Y_train)
plt.plot(X_train, Y_pred, color='red')
plt.show()
