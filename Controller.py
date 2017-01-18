import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import PerceptronService as ps

df = pd.read_csv('iris.data', header=None)
# print df
df.tail()
# print  df

# select setosa and versicolor
y = df.iloc[0:100, 4].values
print "========== setosa and versicolor ==== " + y
y = np.where(y == 'Iris-setosa', -1, 1)
# print y

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
print " ====== sepal and petal lengths ====== "
print X
# plot data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color ='blue', marker ='x', label ='versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

ppn = ps.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_)
plt.xlabel("Epochs")
plt.ylabel("Number of misclassifications")
plt.show()