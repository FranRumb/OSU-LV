import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
unique_colors = np.unique(img_array)
print(len(unique_colors))

# rezultatna slika
img_array_aprox = img_array.copy()

#

km1 = KMeans(n_clusters=1, init='k-means++', n_init=5, random_state=0)
km1.fit(img_array)
km2 = KMeans(n_clusters=2, init='k-means++', n_init=5, random_state=0)
km2.fit(img_array)
km3 = KMeans(n_clusters=3, init='k-means++', n_init=5, random_state=0)
km3.fit(img_array)
km4 = KMeans(n_clusters=4, init='k-means++', n_init=5, random_state=0)
km4.fit(img_array)
km5 = KMeans(n_clusters=5, init='k-means++', n_init=5, random_state=0)
km5.fit(img_array)
km6 = KMeans(n_clusters=6, init='k-means++', n_init=5, random_state=0)
km6.fit(img_array)
km7 = KMeans(n_clusters=7, init='k-means++', n_init=5, random_state=0)
km7.fit(img_array)
km8 = KMeans(n_clusters=8, init='k-means++', n_init=5, random_state=0)
km8.fit(img_array)
km9 = KMeans(n_clusters=9, init='k-means++', n_init=5, random_state=0)
km9.fit(img_array)

x=[1,2,3,4,5,6,7,8,9]
y=[km1.inertia_, km2.inertia_, km3.inertia_, km4.inertia_, km5.inertia_, km6.inertia_, km7.inertia_, km8.inertia_, km9.inertia_]
plt.plot(x,y)
plt.show()

#
labels = km1.predict(img_array)
clusteredColors = km1.cluster_centers_[labels]
for i in range(len(img_array_aprox)):
    img_array_aprox[i] = clusteredColors[i]

plt.title("Rezultatna slika 1")
plt.imshow(img_array_aprox.reshape(w,h,d))
plt.tight_layout()
plt.show()

#
labels = km2.predict(img_array)
clusteredColors = km2.cluster_centers_[labels]
for i in range(len(img_array_aprox)):
    img_array_aprox[i] = clusteredColors[i]

plt.title("Rezultatna slika 2")
plt.imshow(img_array_aprox.reshape(w,h,d))
plt.tight_layout()
plt.show()

#
labels = km3.predict(img_array)
clusteredColors = km3.cluster_centers_[labels]
for i in range(len(img_array_aprox)):
    img_array_aprox[i] = clusteredColors[i]

plt.title("Rezultatna slika 3")
plt.imshow(img_array_aprox.reshape(w,h,d))
plt.tight_layout()
plt.show()

#
labels = km4.predict(img_array)
clusteredColors = km4.cluster_centers_[labels]
for i in range(len(img_array_aprox)):
    img_array_aprox[i] = clusteredColors[i]

plt.title("Rezultatna slika 4")
plt.imshow(img_array_aprox.reshape(w,h,d))
plt.tight_layout()
plt.show()

#
labels = km5.predict(img_array)
clusteredColors = km5.cluster_centers_[labels]
for i in range(len(img_array_aprox)):
    img_array_aprox[i] = clusteredColors[i]

plt.title("Rezultatna slika 5")
plt.imshow(img_array_aprox.reshape(w,h,d))
plt.tight_layout()
plt.show()


#------------------------------------------------------------------------------------
#2
# ucitaj sliku
img = Image.imread("imgs\\test_2.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
unique_colors = np.unique(img_array)
print(len(unique_colors))

# rezultatna slika
img_array_aprox = img_array.copy()

km = KMeans(n_clusters=5, init='k-means++', n_init=5, random_state=0)
km.fit(img_array)
labels = km.predict(img_array)
clusteredColors = km.cluster_centers_[labels]

for i in range(len(img_array_aprox)):
    img_array_aprox[i] = clusteredColors[i]

plt.title("Rezultatna slika")
plt.imshow(img_array_aprox.reshape(w,h,d))
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------
#3
# ucitaj sliku
img = Image.imread("imgs\\test_3.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
unique_colors = np.unique(img_array)
print(len(unique_colors))

# rezultatna slika
img_array_aprox = img_array.copy()

km = KMeans(n_clusters=5, init='k-means++', n_init=5, random_state=0)
km.fit(img_array)
labels = km.predict(img_array)
clusteredColors = km.cluster_centers_[labels]

for i in range(len(img_array_aprox)):
    img_array_aprox[i] = clusteredColors[i]

plt.title("Rezultatna slika")
plt.imshow(img_array_aprox.reshape(w,h,d))
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------
#4
# ucitaj sliku
img = Image.imread("imgs\\test_4.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
unique_colors = np.unique(img_array)
print(len(unique_colors))

# rezultatna slika
img_array_aprox = img_array.copy()

km = KMeans(n_clusters=5, init='k-means++', n_init=5, random_state=0)
km.fit(img_array)
labels = km.predict(img_array)
clusteredColors = km.cluster_centers_[labels]

for i in range(len(img_array_aprox)):
    img_array_aprox[i] = clusteredColors[i]

plt.title("Rezultatna slika")
plt.imshow(img_array_aprox.reshape(w,h,d))
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------
#5
# ucitaj sliku
img = Image.imread("imgs\\test_5.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
unique_colors = np.unique(img_array)
print(len(unique_colors))

# rezultatna slika
img_array_aprox = img_array.copy()

km = KMeans(n_clusters=5, init='k-means++', n_init=5, random_state=0)
km.fit(img_array)
labels = km.predict(img_array)
clusteredColors = km.cluster_centers_[labels]

for i in range(len(img_array_aprox)):
    img_array_aprox[i] = clusteredColors[i]

plt.title("Rezultatna slika")
plt.imshow(img_array_aprox.reshape(w,h,d))
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------
#6
# ucitaj sliku
img = Image.imread("imgs\\test_6.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
unique_colors = np.unique(img_array)
print(len(unique_colors))

# rezultatna slika
img_array_aprox = img_array.copy()

km = KMeans(n_clusters=5, init='k-means++', n_init=5, random_state=0)
km.fit(img_array)
labels = km.predict(img_array)
clusteredColors = km.cluster_centers_[labels]

for i in range(len(img_array_aprox)):
    img_array_aprox[i] = clusteredColors[i]

plt.title("Rezultatna slika")
plt.imshow(img_array_aprox.reshape(w,h,d))
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------
#Binarni
# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
unique_colors = np.unique(img_array)
print(len(unique_colors))

# rezultatna slika
img_array_aprox = img_array.copy()

km = KMeans(n_clusters=5, init='k-means++', n_init=5, random_state=0)
km.fit(img_array)
labels = km.predict(img_array)
clusteredColors = km.cluster_centers_[labels]

for i in km.cluster_centers_:
    for j in range(len(img_array_aprox)):
        if((i == clusteredColors[j]).any()):
            img_array_aprox[j] = [1,1,1]
        else:
            img_array_aprox[j] = [0,0,0]
    plt.title("Binarna slika")
    plt.imshow(img_array_aprox.reshape(w,h,d))
    plt.tight_layout()
    plt.show()

