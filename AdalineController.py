import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import AdaptiveLinearNeuron as adalineGD
import Plotter as plot

df = pd.read_csv('iris.data', header=None)
# print df
df.tail()
# print  df

# select setosa and versicolor
y = df.iloc[0:100, 4].values
# print "========== setosa and versicolor ==== " + y
y = np.where(y == 'Iris-setosa', -1, 1)
# print y

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
# print " ====== sepal and petal lengths ====== "
# print X

# fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 4))
# ada1 = adalineGD.Adaline(eta = 0.01, n_iter = 10, ).fit(X,y)
# ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('log(Sum-squared-error')
# ax[0].set_title('Adaline - learning rate 0.01')
#
# ada2 = adalineGD.Adaline(eta = 0.0001, n_iter = 10, ).fit(X,y)
# ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('log(Sum-squared-error')
# ax[1].set_title('Adaline - learning rate 0.01')
# # plt.show()

# standardize features

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = adalineGD.Adaline(n_iter=15, eta=0.01)
ada.fit(X_std, y)

plot.plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./adaline_2.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')

plt.tight_layout()
# plt.savefig('./adaline_3.png', dpi=300)
plt.show()








