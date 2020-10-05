## Dataset : https://archive.ics.uci.edu/ml/datasets/student%2Bperformance
## Video : https://techwithtim.net/tutorials/machine-learning-python/linear-regression/


import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


"""STEP1"""
#Reading students data csv
data = pd.read_csv("C:\SelfDownload\RegressionAlgos\student_mat.csv", sep=";")
#print(data.head())


"""STEP2"""
#Choosing the features need to be used for predction (List of attributes from the data set)
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]


"""Label that needs to be predcited"""
predict = "G3"

"""STEP3"""
#Choosing the features need to be used for predction (List of attributes from the data set)
X = np.array(data.drop([predict], 1))
# print("The value of x is \n", X)

"""STEP4"""
#Define label (List of attributes from the data set that needs to be predicted)
y = np.array(data[predict])
# print("The value of y is \n", y)


##------------------------------------------------------------------------------------------------##
""" 
    We want to achieve the best accuracy, thats why iterating over the below logic to attain the 
    best results. This needs to be done only once, because once the model is trained then we can 
    comment out the below code (from STEP5 till STEP8) and work.
"""

best = 0
for _ in range(30):
  """STEP5"""
  #Splitting training and testing data using train_test_split library
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)


  """STEP6"""
  #Initializing linear regression model
  model = linear_model.LinearRegression()

  """STEP7"""
  #Fitiing training data into model (Training the data set here)
  model.fit(x_train, y_train)

  """"Additional Calculations"""
  #showing the score (accurancy of the predcition model)
  accuracy = model.score(x_test, y_test)
  print('Accuracy (Score)  \n',  accuracy)


  """ Saving the model using pickle """

  if accuracy > best:
    best == accuracy
    """STEP8"""
    ##Saves the pickle file in the directory (Pickle just saves the model)
    with open("studentmodel.pickle", "wb") as f:
      pickle.dump(model, f)
##------------------------------------------------------------------------------------------------##


"""STEP9"""
##open a file saved by pickle
pickle_in = open("studentmodel.pickle", 'rb')

"""STEP10"""
##loading pickle in linear model
linear = pickle.load(pickle_in)


"""STEP11"""
#Predicting the testing data set
predictions = linear.predict(x_test)

##Printing all the predictions,with the given input data set
for x in range(len(predictions)):
  print(predictions[x], x_test[x], y_test[x])


"""STEP12"""
##Plotting the graph

"""This p below is the x axis of the graph (which means we can pick any values from the STEP2 except
   "G3" becasue that's thats our lable or our Y axis of the graph) """

p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
