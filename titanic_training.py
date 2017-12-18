import pandas as pd
import numpy as np
from sklearn import svm

# Config
pd.options.mode.chained_assignment = None

# Read 'train' and 'test' csv files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#CLEANING DATA

# Update missing age with median value
median_age = train["Age"].median()
train["Age"] = train["Age"].fillna(median_age)
test["Age"] = test["Age"].fillna(median_age)

# Add Child feature -> Age < 18
train["Child"] = 0
train["Child"][train["Age"] < 18] = 1
test["Child"] = 0
test["Child"][test["Age"] < 18] = 1

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2


# TRAIN using SVM - Gaussian Kernel (Radial basis function)
svm_gaussian = svm.SVC(kernel="rbf")

X_train = train[["Pclass", "Age", "SibSp", "Parch", "Child", "Embarked"]]
y_train = train["Survived"]

X_test = test[["Pclass", "Age", "SibSp", "Parch", "Child", "Embarked"]]

svm_gaussian.fit(X_train, y_train)
y_test = svm_gaussian.predict(X_test)


# Write output
# Create a data frame with two columns: PassengerId & Survived
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(y_test, PassengerId, columns=["Survived"])
prediction_file = my_solution.to_csv(path_or_buf="prediction.csv", index_label="PassengerId")