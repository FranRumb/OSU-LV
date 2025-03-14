import numpy as np
import matplotlib.pyplot as plt

ones = np.ones((50, 50))
ones = ones * 255
zeros = np.zeros((50, 50))

topPart = np.hstack((zeros, ones))
img = np.vstack((topPart, np.fliplr(topPart)))

plt.figure()
plt.imshow(img , cmap ="gray")
plt.show()