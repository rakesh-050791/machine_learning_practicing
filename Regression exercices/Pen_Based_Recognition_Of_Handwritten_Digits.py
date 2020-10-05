## Source :  https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html?highlight=digit%20dataset
## Dataset : https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
## Video :   https://www.youtube.com/watch?v=J5bXOOmkopc


import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn


"""STEP1"""
## Load training data set using sklearn, load digits method
digits = load_digits()

"""Exploring the digits direcrtory"""
dir(digits)
# print(dir(digits)) : Output = ['DESCR', 'data', 'images', 'target', 'target_names']
# print(digits.data[0]) : Output =  one dimensional array (8*8 elements)


"""Exploring and printing the actual images from the data"""
# plt.gray()
# for i in range(5):
#   plt.matshow(digits.images[i])
#   plt.show()


"""Taget contains the actual number printed on the image in above step"""
# print(digits.target[0:5])

"""STEP2"""
""" Now we can use Data and Target to train our model"""
#Splitting training and testing data using train_test_split library
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
# print('X Train  \n',len(X_train))

"""STEP4"""
#Initializing logistic regression model
model = LogisticRegression()


"""STEP5"""
#Fitiing training data into model (Training the data set here)
model.fit(X_train, y_train)


"""Pickup a random sample"""
#plt.matshow(digits.images[63])
#plt.show()
#print('Target No.   \n',  digits.target[63])


"""STEP6"""
#Predicting the testing data set
y_predicted = model.predict(X_test)            #  --predicting all test values
# prediction = model.predict([digits.data[63]])  --predicting particular no
prediction = model.predict(digits.data[0:5])  #  --predicting range
print('PREDICTION   \n', prediction)



"""Confusion Matrix"""
## To identify where our model didn't do well (as we are not getting 100% accuracy or 100% score)
 #Which is the truth --y_test
 #Which is the predicted value --y_predicted
cm = confusion_matrix(y_test, y_predicted)
print('Confusion Matrix   \n', cm)


"""Confusion Matrix Visualization"""
plt.figure(figsize= (10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()



""""Additional Calculations"""
#showing the score (accurancy of the predcition model)
# model.score(X_test, y_test)
print('SCORE   \n', model.score(X_test, y_test))



