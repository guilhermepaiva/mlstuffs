import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def compute_knn():

  """
    this function predict the class for the two tests (first_test and second_test).
    for more information take a look at www.cin.ufpe.br/~if699 (machine learning class at CIn/UFPE). this algorithms are just some pratices from the lectures.
  """

  a = np.matrix('50 1.6 1; 53 1.65 1; 60 1.58 1; 62 1.62 1; 91 1.75 2; 102 1.85 2; 105 1.82 2; 103 1.77 2; 87 1.73 2') #the dataset
  print "This is the complete dataset: " + "\n"
  print a
  print "------------------------------------------------"
  first_test = np.matrix('70 1.63')
  second_test = np.matrix('83 1.77')
  
  X = a[:, :2] #get the first two features
  y = a[:, 2] #get the target values
  
  neigh = KNeighborsClassifier(n_neighbors=3) #redefine the number of neighbors, default is 5
  neigh.fit(X, y) #fit the model using X as training data and y as target values
  
  print "The first test is: "
  print first_test
  print "...and it's classified as: "
  print neigh.predict(first_test)
  print "------------------------------------------------"
  print "The second test is: "
  print second_test
  print "...and it's classified as: "
  print neigh.predict(second_test)

if __name__ == "__main__":
  compute_knn()

  

