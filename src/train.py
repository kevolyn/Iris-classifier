#load data for this project, Iris Project
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
#use joblib to save file
import joblib



iris=load_iris()
X=iris.data #shape(150,4)
y=iris.target #(150,)
print(iris.feature_names, iris.target_names)


#Bring out X and y parameters, and place data into 80% for prediction values, and 20% (0.2) as true labels for testing against (y_test))
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)



#use decisiontreeclassifier model to classify data.
model = DecisionTreeClassifier(random_state=42)


#train model to fit data into sub-categories for classification
model.fit( X_train, y_train)


#use model.predict to bring an array of predicted data. use y_pred for ilustration.
y_pred = model.predict(X_test)


#Accuracy metric
accuracy = accuracy_score( y_test, y_pred)
print("accuracy:", accuracy)
#the closest you are to 1.0, the more accurate


#Accuracy model- Confusion matrix
print(confusion_matrix( y_test, y_pred))
#perfect matrix will appear as
# [10  0  0 ]
# [0  10  0 ]
# [0  0  10 ]



cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot().figure_.savefig('confusion_matrix.png')



joblib.dump(model, "model.joblib")
