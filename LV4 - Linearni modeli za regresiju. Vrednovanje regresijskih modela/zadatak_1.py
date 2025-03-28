import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sklearn.linear_model as lm

data = pd.read_csv("data_C02_emission.csv")
#a)

X = data[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)']]
y = data['CO2 Emissions (g/km)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#b)
plt.figure()
plt.scatter(x=y_train,y=X_train['Engine Size (L)'], c='blue')
plt.scatter(x=y_test, y=X_test['Engine Size (L)'], c='red')
plt.show()

#c)

sc = MinMaxScaler()

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
plt.hist(X_train, bins=10)
plt.show()
plt.hist(X_train_scaled, bins=10)
plt.show()

#d)

linearModel = lm.LinearRegression()
linearModel.fit(X_train_scaled, y_train)

print(linearModel.coef_)

#e)

y_test_p = linearModel.predict(X_test_scaled)
MAE = mean_absolute_error(y_pred= y_test_p, y_true=y_test)
MSE = mean_squared_error(y_pred= y_test_p, y_true=y_test)
r2 = r2_score(y_pred= y_test_p, y_true=y_test)
print(MAE)
print(MSE)
print(r2)
plt.scatter(x=y_test_p, y=y_test, s=10)
plt.show()

