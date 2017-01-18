from sklearn.tree import DecisionTreeClassifier
from sklearn import  datasets
import numpy as np
import DecistionRegionPlotter as dcplot
import  matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Splitting data into 70% training and 30% test data:
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3,random_state=0)
tree.fit(X_train, Y_train)
X_combined = np.vstack((X_train, X_test))
Y_combined = np.hstack((Y_train,Y_test))
dcplot.plot_decision_regions(X_combined, Y_combined, classifier=tree, test_idx=range(105,150))
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc = 'upper left')
plt.show()

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file='tree.dot', feature_names=['petal length', 'petal width'])