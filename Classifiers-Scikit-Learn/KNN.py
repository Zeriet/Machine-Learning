from sklearn.linear_model import  LogisticRegression
from sklearn import datasets
import DecistionRegionPlotter as dp
import numpy as np
import matplotlib.pyplot as plt
import  DecistionRegionPlotter as dp
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print(' Class labels:', np.unique(y))


# Added version check for recent scikit-learn 0.18 checks
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version

# Splitting data into 70% training and 30% test data:

if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train, Y_train)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((Y_train, Y_test))

dp.plot_decision_regions(X_combined_std, y_combined,
                      classifier=knn, test_idx=range(105, 150))

from sklearn.preprocessing import StandardScaler



plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/k_nearest_neighbors.png', dpi=300)
plt.show()
