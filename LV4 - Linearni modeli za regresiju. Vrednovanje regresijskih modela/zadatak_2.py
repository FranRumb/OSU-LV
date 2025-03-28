import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from sklearn.metrics import r2_score
import sklearn.linear_model as lm
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("data_C02_emission.csv")

X = data[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)']]
y = data['CO2 Emissions (g/km)']

ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()
X['Fuel Type'] = X_encoded

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)
y_test_p = linearModel.predict(X_test)

r2 = r2_score(y_pred=y_test_p, y_true=y_test)
MAE = mean_absolute_error(y_pred=y_test_p, y_true=y_test)
print(MAE)
print(r2)
print(max_error(y_true=y_test, y_pred=y_test_p))

plt.scatter(x=y_test_p, y=y_test, s=10)
plt.show()

difference = abs(y_test_p - y_test)
error = np.argmax(difference)
print(error)
print(data.iloc[error,1])

