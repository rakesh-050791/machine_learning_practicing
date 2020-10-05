## Dataset : https://archive.ics.uci.edu/ml/datasets/student%2Bperformance
## Video : https://techwithtim.net/tutorials/machine-learning-python/linear-regression/


import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model


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


"""STEP5"""
#Splitting training and testing data using train_test_split library
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)


"""STEP6"""
#Initializing linear regression model
model = linear_model.LinearRegression()


"""STEP7"""
#Fitiing training data into model (Training the data set here)
model.fit(x_train, y_train)


"""STEP8"""
#Predicting the testing data set
predictions = model.predict(x_test)

##Printing all the predictions,with the given input data set
for x in range(len(predictions)):
  print(predictions[x], x_test[x], y_test[x])


""""Additional Calculations"""
#showing the score (accurancy of the predcition model)
accuracy = model.score(x_test, y_test)
print('Accuracy (Score)  \n',  accuracy)

#model.coef_ indicates value of m in y=m*x + b equation
print('Coefficient  \n',  model.coef_)

#model.intercept_ indicates value of b in y=m*x + b equation
print('Intercept  \n',  model.intercept_)