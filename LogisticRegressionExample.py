## Dataset : https://github.com/codebasics/py/blob/master/ML/7_logistic_reg/insurance_data.csv
## Video : https://www.youtube.com/watch?v=zM4VZR0px8E


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


"""STEP1"""
#Reading students data csv
df = pd.read_csv("C:\SelfDownload\RegressionAlgos\insurance_data.csv", sep=";")
df.head()

"""STEP2"""
#Plotting a scattered graph with students data to analyze
plt.scatter(df.age, df.bought_insurance, marker='+', color='red')
# plt.show()


"""STEP3"""
#Splitting training and testing data using train_test_split library
X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, test_size=0.1)
print(X_test)
# print(y_test)

"""STEP4"""
#Initializing logistic regression model
model = LogisticRegression()

"""STEP5"""
#Fitiing training data into model (Training the data set here)
model.fit(X_train, y_train)

"""STEP6"""
#Predicting the testing data set
prediction = model.predict(X_test)
print(prediction)


#Predicting the testing data set (In percentage %)
prediction_probability = model.predict_proba(X_test)
print(prediction_probability)

""""Additional Calculations"""

#showing the score (accurancy of the predcition model)
print(model.score(X_test, y_test))


#model.coef_ indicates value of m in y=m*x + b equation
print('Coefficient  \n',model.coef_)


#model.intercept_ indicates value of b in y=m*x + b equation
print('Intercept  \n',model.intercept_)