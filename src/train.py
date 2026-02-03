#load data for this project, Iris Project
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data #shape(150,4)
y=iris.target #(150,)
print(iris.feature_names, iris.target_names)