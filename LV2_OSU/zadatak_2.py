import numpy as np
import matplotlib.pyplot as plt
import csv

with open("data.csv", "r") as f:
    next(f)
    reader = csv.reader(f)
    data_string = list(reader)

data = np.array(data_string, float)

#a)
print(data.shape)

#b)
plt.scatter(data[0:10001, 1], data[0:10001, 2], 1, 'b', '.')
plt.xlabel("Visina/cm")
plt.ylabel("Masa/kg")
plt.title("Scatter visina/masa")
plt.show()

#c)
plt.scatter(data[0::50, 1], data[0::50, 2], 1, 'b', '.')
plt.xlabel("Visina/cm")
plt.ylabel("Masa/kg")
plt.title("Scatter visina/masa")
plt.show()

#d)
print("Min:", np.min(data[:, 1]))
print("Max:", np.max(data[:, 1]))
print("Mean:", np.mean(data[:, 1]))

#e)
maleData = data[np.isin(data[:, 0], 1)]

print("Male min:", np.min(maleData[:, 1]))
print("Male max:", np.max(maleData[:, 1]))
print("Male mean:", np.mean(maleData[:, 1]))

femaleData = data[np.isin(data[:, 0], 0)]

print("Female min:", np.min(femaleData[:, 1]))
print("Female max:", np.max(femaleData[:, 1]))
print("Female mean:", np.mean(femaleData[:, 1]))