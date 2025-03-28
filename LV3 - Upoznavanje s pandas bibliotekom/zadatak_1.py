import pandas as pd
import numpy as np

data = pd.read_csv("data_C02_emission.csv")

#a)

print("a)")
print("Size:")
print(len(data))
print("\n")
print("Info:")
print(data.info())
print("\n")
print("Description:")
print(data.describe())
print("\n")
print("Nulls:")
print(data.isnull().sum())
print("\n")
print("Duplicates:")
print(data.duplicated().sum())
print("\n")
data.drop_duplicates()
data.dropna(axis=0)

#b)

print("b)")
print(data.sort_values(by=["Fuel Consumption City (L/100km)"]).tail(3).iloc[:, [0, 1, 7]])
print(data.sort_values(by=["Fuel Consumption City (L/100km)"]).head(3).iloc[:, [0, 1, 7]])
print("\n")

#c)

print("c)")
newData = data[(data['Engine Size (L)'] > 2.5)]
newData = newData[(newData['Engine Size (L)']) < 3.5]
print(len(newData))
print(newData['CO2 Emissions (g/km)'].mean())
print("\n")

#d)

print("d)")
newData = data[(data['Make'] == "Audi")]
print(len(newData))
newData = newData[(newData['Cylinders'] == 4)]
print(newData["CO2 Emissions (g/km)"].mean())
print("\n")

#e)

print("e)")
fourCylinderData = data[(data['Cylinders'] == 4)]
print("Four cylinder cars:")
print(len(fourCylinderData))
print(fourCylinderData["CO2 Emissions (g/km)"].mean())
print("\n")

sixCylinderData = data[(data['Cylinders'] == 6)]
print("Six cylinder cars:")
print(len(sixCylinderData))
print(sixCylinderData["CO2 Emissions (g/km)"].mean())
print("\n")

eightCylinderData = data[(data['Cylinders'] == 8)]
print("Eight cylinder cars:")
print(len(eightCylinderData))
print(eightCylinderData["CO2 Emissions (g/km)"].mean())
print("\n")

twelveCylinderData = data[(data['Cylinders'] == 12)]
print("Twelve cylinder cars:")
print(len(twelveCylinderData))
print(twelveCylinderData["CO2 Emissions (g/km)"].mean())
print("\n")

sixteenCylinderData = data[(data['Cylinders'] == 16)]
print("Sixteen cylinder cars:")
print(len(sixteenCylinderData))
print(sixteenCylinderData["CO2 Emissions (g/km)"].mean())
print("\n")

#f)

print("f)")

dieselConsumptionData = data[(data["Fuel Type"] == "D")]
print("Diesel Car City Consumption")
print(dieselConsumptionData['Fuel Consumption City (L/100km)'].mean())
print(dieselConsumptionData['Fuel Consumption City (L/100km)'].median())
print("\n")

petrolConsumptionData = data[(data["Fuel Type"] == "X")]
print("Petrol Car City Consumption")
print(petrolConsumptionData['Fuel Consumption City (L/100km)'].mean())
print(petrolConsumptionData['Fuel Consumption City (L/100km)'].median())
print("\n")

#g)

print("g)")

print(data[(data["Cylinders"] == 4) & (data["Fuel Type"] == "D")].max())
print("\n")

#h)

print("h)")

manualTransmissionData = pd.DataFrame(
    data = {'type': ["M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"]}
)
print(len(data[(data['Transmission'].isin(manualTransmissionData['type']))]))
print("\n")

#i)

print("i)")
print(data.corr(numeric_only=True))