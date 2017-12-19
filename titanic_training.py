import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import svm
from sklearn import model_selection as ms

# Config
pd.options.mode.chained_assignment = None

# Read 'train' and 'test' csv files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train_test_data = [train, test]


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

def facet(feature):
    facet = sns.FacetGrid(train, hue="Survived", aspect=4)
    facet.map(sns.kdeplot, feature, shade=True)
    facet.set(xlim=(0, train[feature].max()))
    facet.add_legend()


def visualize_data():

    bar_chart("Sex")        # Women -> more chance to survive
    bar_chart("Pclass")     # 1st class -> more chance to survive; 3rd class -> more chance to die
    bar_chart("SibSp")      # If you have 1 Sibling or Spouse -> more chance to survive
    bar_chart('Parch')      # If you have 1 Parent/Child -> more chance to survive
    bar_chart('Embarked')   # C -> maybe more chance to survive?

    facet('Age')
    facet('Fare')

    Pclass1 = train[train['Pclass'] == 1]['Embarked'].value_counts()
    Pclass2 = train[train['Pclass'] == 2]['Embarked'].value_counts()
    Pclass3 = train[train['Pclass'] == 3]['Embarked'].value_counts()
    df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
    df.index = ['1st class', '2nd class', '3rd class']
    df.plot(kind='bar', stacked=True, figsize=(10,5))

    plt.show()

def compute_title():
    for dataset in train_test_data:
        dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    title_mapping = {
        "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3,
        "Dr": 4, "Rev": 4, "Major": 4, "Col": 4, "Mlle": 4, "Jonkheer": 4,
        "Countess": 4, "Ms": 4, "Capt": 4, "Lady": 4, "Sir": 4, "Don": 4, "Dona": 4, "Mme": 4
    }

    for dataset in train_test_data:
        dataset['Title'] = dataset["Title"].map(title_mapping)


def compute_sex():

    sex_mapping = { "male": 0, "female": 1 }
    for dataset in train_test_data:
        dataset['Sex'] = dataset['Sex'].map(sex_mapping)


def compute_age():
    # Fill missing values with median of title
    train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)
    test['Age'].fillna(test.groupby('Title')['Age'].transform('median'), inplace=True)

    for dataset in train_test_data:
        dataset.loc[ dataset['Age'] <= 18, 'Age'] = 0, # child
        dataset.loc[ (dataset['Age'] > 18) & (dataset['Age'] <= 35), 'Age'] = 1, # young
        dataset.loc[ (dataset['Age'] > 35) & (dataset['Age'] <= 45), 'Age'] = 2, # adult
        dataset.loc[ (dataset['Age'] > 45) & (dataset['Age'] <= 59), 'Age'] = 3, # mid-aged
        dataset.loc[ dataset['Age'] > 59, 'Age'] = 4, # senior

def compute_embarked():
    for dataset in train_test_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')

    embarked_mapping = { "S": 0, "C": 1, "Q": 2 }
    for dataset in train_test_data:
        dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


def compute_fare():

    train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'), inplace=True)
    test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'), inplace=True)

    for dataset in train_test_data:
        dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0, # very cheap
        dataset.loc[ (dataset['Fare'] > 17) & (dataset['Fare'] <= 29), 'Fare'] = 1, # cheap
        dataset.loc[ (dataset['Fare'] > 29) & (dataset['Fare'] <= 100), 'Fare'] = 2, # expensive
        dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3 # very expensive

def compute_family():
    for dataset in train_test_data:
        dataset['FamilySize'] = (dataset['SibSp'] + dataset['Parch'] + 1) * 0.1 # + feature scaling


def drop_extra_features():
    features_drop = ['Ticket', 'SibSp', 'Parch', 'Cabin', 'Name']
    train.drop(features_drop, axis=1, inplace=True)
    test.drop(features_drop, axis=1, inplace=True)
    train.drop('PassengerId', axis=1, inplace=True)


def feature_engineering():
    compute_title()
    compute_sex()
    compute_age()
    compute_embarked()
    compute_fare()
    compute_family()
    drop_extra_features()


def cross_validation():
    # TRAIN using SVM - Gaussian Kernel (Radial basis function)
    svm_gaussian = svm.SVC(kernel="rbf")

    X = train[["Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "FamilySize"]]
    y = train["Survived"]

    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=0)

    classifier = svm_gaussian.fit(X_train, y_train)
    cv_score = classifier.score(X_test, y_test)
    print(cv_score)


def svm_train():
    # TRAIN using SVM - Gaussian Kernel (Radial basis function)
    svm_gaussian = svm.SVC(kernel="rbf")
    X = train[["Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "FamilySize"]]
    y = train["Survived"]
    classifier = svm_gaussian.fit(X, y)
    X_test = test[["Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "FamilySize"]]
    y_test = classifier.predict(X_test)

    return y_test


def output_result(y_test):
    # Write output
    # Create a data frame with two columns: PassengerId & Survived
    PassengerId = np.array(test["PassengerId"]).astype(int)
    my_solution = pd.DataFrame(y_test, PassengerId, columns=["Survived"])
    my_solution.to_csv(path_or_buf="prediction.csv", index_label="PassengerId")


# explore_data()
feature_engineering()
# visualize_data()
# cross_validation()
y_test = svm_train()
output_result(y_test)
