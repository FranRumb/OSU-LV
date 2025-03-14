import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
img = img [:,:,0].copy()

#a)
brightImg = img * 2

plt.figure()
plt.imshow(brightImg , cmap ="gray")
plt.show()

#b)
quarterSize = img.shape[1]/4
quarterSize = int(quarterSize)

quarterImg = img[:, quarterSize:2*quarterSize]

plt.figure()
plt.imshow(quarterImg , cmap ="gray")
plt.show()

#c)
rotatedImg = np.rot90(img)

plt.figure()
plt.imshow(rotatedImg , cmap ="gray")
plt.show()

#d)
mirroredImg = np.fliplr(img)

plt.figure()
plt.imshow(mirroredImg , cmap ="gray")
plt.show()