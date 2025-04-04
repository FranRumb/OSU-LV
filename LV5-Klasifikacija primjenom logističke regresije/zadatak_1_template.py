import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


#a)

colors = np.array(["red", "blue"])

plt.figure()
plt.scatter(x=X_train[: , 0], y=X_train[: , 1],c=y_train)
plt.scatter(x=X_test[: , 0], y=X_test[: , 1], marker='x', c=colors[y_test])
plt.show()

#b)

logRegressionModel = LogisticRegression()
logRegressionModel.fit(X=X_train, y=y_train)

#c)
X1_plot_points = np.linspace(-4,3,100)
coefs = logRegressionModel.coef_
free_coef = logRegressionModel.intercept_
y = -(coefs[0,0]/coefs[0,1])*X1_plot_points - free_coef/coefs[0,1]
plt.scatter(x=X_test[: , 0], y=X_test[: , 1], c=colors[y_test])
plt.plot(X1_plot_points, y, '-r')
plt.show()

#d)

y_test_p = logRegressionModel.predict(X_test)
cm = confusion_matrix(y_true=y_test, y_pred=y_test_p,labels=logRegressionModel.classes_)
disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logRegressionModel.classes_)
disp_cm.plot()
plt.show()
print(classification_report(y_true=y_test, y_pred=y_test_p))

#e)

colors = np.array(["green", "black"])
y_r_w = abs(y_test-y_test_p)
plt.scatter(x=X_test[:, 0], y=X_test[:, 1], c=colors[y_r_w])
plt.show()