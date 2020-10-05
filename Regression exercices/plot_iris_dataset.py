##Source : https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
##Dataset : https://archive.ics.uci.edu/ml/datasets/iris


from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

"""STEP1"""
## Load training data set using sklearn, load digits method
iris = load_iris()
# breakpoint()


"""Exploring the digits direcrtory"""
dir(iris)
# print(dir(iris)) #: Output = ['DESCR', 'data', 'images', 'target', 'target_names']
# print(iris.data[0]) #: Output =  one dimensional array (2*2 elements) e.g [5.1 3.5 1.4 0.2]

X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
