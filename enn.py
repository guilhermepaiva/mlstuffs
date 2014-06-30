import numpy as np
from sklearn import neighbors
from sklearn import datasets
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors.classification import KNeighborsClassifier

def compute_enn(X, y):
  """
  the edited nearest neighbors removes the instances in the boundaries, maintaining reduntant samples
  """

  classifier = KNeighborsClassifier(n_neighbors=3)

  classes = np.unique(y)
  classes_ = classes

  mask = np.zeros(y.size, dtype=bool)
  classifier.fit(X, y)

  for i in xrange(y.size):
    sample, label = X[i], y[i]
    if classifier.predict(sample) == [label]:
      mask[i] = not mask[i]

  X_ = np.asarray(X[mask])
  y_ = np.asarray(y[mask])
  reduction_ = 1.0 - float(len(y_)) / len(y)
  print reduction_


if __name__ == "__main__":
  iris = datasets.load_iris()
  X = iris.data
  y = iris.target

  compute_enn(X, y)
