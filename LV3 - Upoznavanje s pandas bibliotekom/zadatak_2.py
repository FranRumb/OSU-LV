import cmath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col

data = pd.read_csv("data_C02_emission.csv")

#a) 

plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist', bins = 10)
plt.show()

#b)



colors = []
data.plot.scatter(
    x = 'Fuel Consumption City (L/100km)',
    y = 'CO2 Emissions (g/km)',
    s = 20
)
plt.show()

#c)

data.boxplot(column=['Fuel Consumption Hwy (L/100km)'], by='Fuel Type')
plt.show()

#d)

fuelTypes = ['X', 'Z', 'D', 'E']
plt.bar(fuelTypes, data.groupby('Fuel Type')['Make'].count().to_list())
plt.show()

#e)

cylinderCounts = [1,2,3,4,6,8,12,16]
cylinderGroupedData = data.groupby('Cylinders')
plt.bar(cylinderCounts, cylinderGroupedData['CO2 Emissions (g/km)'].mean())
plt.show()