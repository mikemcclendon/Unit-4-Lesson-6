from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets
from matplotlib.colors import ListedColormap
import numpy as np

#setting up iris feature data for setosa and versicolor (first 100 lines)
iris = datasets.load_iris()
feature1 = iris.data[0:100, 0]
feature2 = iris.data[0:100, 1]
feature3 = iris.data[0:100, 2]
feature4 = iris.data[0:100, 3]
y = iris.target[0:100] 
#setting up data to classify three flowers, setosa, versicolor, and virginica (first 150)
multifeature1 = iris.data[0:150, 0]
multifeature2 = iris.data[0:150, 1]
ymulti = iris.target[0:150]


#setting estimators for a lot of charts as well as the pairwise combo of all the features
#and one combination of three flowers
svc1 = svm.SVC(kernel='linear')
X1 = np.array(zip(feature1, feature2)) #data is range 0-4
svc1.fit(X1, y)
svc2 = svm.SVC(kernel='linear')
X2 = np.array(zip(feature2, feature3))
svc2.fit(X2, y)
svc3 = svm.SVC(kernel='linear')
X3 = np.array(zip(feature3, feature4))
svc3.fit(X3, y)
svc4 = svm.SVC(kernel='linear')
X4 = np.array(zip(feature1, feature4))
svc4.fit(X4, y)
svc5 = svm.SVC(kernel='linear')
X5 = np.array(zip(feature1, feature3))
svc5.fit(X5, y)
svc6 = svm.SVC(kernel='linear')
X6 = np.array(zip(feature2, feature4))
svc6.fit(X6, y)
svcMulti = svm.SVC(kernel='linear')
Xmulti = np.array(zip(multifeature1, multifeature2))
svcMulti.fit(Xmulti, ymulti)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

#plotting all the pairwise feature combos
plot_estimator(svc1, X1, y)
plot_estimator(svc2, X2, y)
plot_estimator(svc3, X3, y)
plot_estimator(svc4, X4, y)
plot_estimator(svc5, X5, y)
plot_estimator(svc6, X6, y)
#plotting the triple combination - doesn't work as well
plot_estimator(svcMulti, Xmulti, ymulti)