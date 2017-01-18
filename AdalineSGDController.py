import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import AdalineStochasticGD as adaSGD
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


X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = adaSGD.AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)

plot.plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochahstic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

plt.tight_layout()
# plt.savefig('./adaline_3.png', dpi=300)
plt.show()








