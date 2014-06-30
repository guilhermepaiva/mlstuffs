import numpy as np
from sklearn import neighbors
from sklearn import datasets
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors.classification import KNeighborsClassifier

def compute_cnn(X, y):

  "condenced nearest neighbor. the cnn removes reduntant instances, maintaining the samples in the decision boundaries."

  classifier = KNeighborsClassifier(n_neighbors=3)

  prots_s = []
  labels_s = []

  classes = np.unique(y)
  classes_ = classes

  for cur_class in classes:
    mask = y == cur_class
    insts = X[mask]
    prots_s = prots_s + [insts[np.random.randint(0, insts.shape[0])]]
    labels_s = labels_s + [cur_class]
    
  classifier.fit(prots_s, labels_s)
  for sample, label in zip(X, y):
    if classifier.predict(sample) != [label]:
      prots_s = prots_s + [sample]
      labels_s = labels_s + [label]
      classifier.fit(prots_s, labels_s)

  X_ = np.asarray(prots_s)
  y_ = np.asarray(labels_s)
  reduction_ = 1.0 - float(len(y_)/len(y))
  print reduction_


if __name__ == "__main__":
  iris = datasets.load_iris()
  X = iris.data
  y = iris.target

  compute_cnn(X, y)
