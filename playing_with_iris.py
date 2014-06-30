import numpy as np
from sklearn import neighbors
from sklearn import datasets
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

def handle():
  #import some data to play with
  iris = datasets.load_iris()
  X, y = iris.data, iris.target

  X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

  #normalized the features
  X_normalized = preprocessing.normalize(X, norm='l2')
  X_train_normalized = preprocessing.normalize(X_train, norm='l2')
  X_test_normalized = preprocessing.normalize(X_test, norm='l2')

  knn = neighbors.KNeighborsClassifier(n_neighbors=3)
  knn.fit(X_train, y_train)
  predicted = knn.predict(X_test)

  print "Classification Report: "
  print metrics.classification_report(y_test, predicted)
  print "\nConfusion Matrix: "
  print metrics.confusion_matrix(y_test, predicted)
  print "\nScores: "
  scores = cross_val_score(knn, iris.data, iris.target, cv=5)
  print scores



if __name__ == "__main__":
  handle()

