import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import model_selection as ms

# Config
pd.options.mode.chained_assignment = None

# Read 'train' and 'test' csv files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


def explore_data():

    print("Train Data Preview")
    print(train.head()) # prints first 5 rows of training data set

    print("\nTrain vs Test Data")
    print(train.shape) # 891 entries, 12 features
    print(test.shape) # 418 entries, 11 features

    print("\nTrain Data Info")
    print(train.info())
    print(train.isnull().sum()) # -> Missing values: Age(177), Cabin(687), Embarked(2)
    print("\nMissing Values")

    print("\nTest Data Info")
    print(test.info())
    print("\nMissing Values")
    print(test.isnull().sum()) # -> Missing values: Age(86), Fare(1), Cabin(327)


def bar_chart(feature):

    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5), title=feature)


def visualize_data():

    bar_chart("Sex")        # Women -> more chance to survive
    bar_chart("Age")        # Too many values
    bar_chart("Pclass")     # 1st class -> more chance to survive; 3rd class -> more chance to die
    bar_chart("SibSp")      # If you have 1 Sibling or Spouse -> more chance to survive
    bar_chart('Parch')      # If you have 1 Parent/Child -> more chance to survive
    bar_chart('Embarked')   # C -> maybe more chance to survive?

    plt.show()



# explore_data()
# visualize_data()



'''

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

X = train[["Pclass", "Age", "SibSp", "Parch", "Child", "Embarked"]]
y = train["Survived"]

test_features = test[["Pclass", "Age", "SibSp", "Parch", "Child", "Embarked"]]

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=0)

classifier = svm_gaussian.fit(X_train, y_train)
cv_score = classifier.score(X_test, y_test)
print(cv_score)

# Write output
# Create a data frame with two columns: PassengerId & Survived
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(y_test, PassengerId, columns=["Survived"])
prediction_file = my_solution.to_csv(path_or_buf="prediction.csv", index_label="PassengerId")

'''