from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

names = ['workclass', 'education', 'marital_status', 'occupation',
                 'relationship', 'race', 'sex', 'native_country', 'salaryLevel']
train = pd.read_csv('./adult.data', sep=',', quotechar='"', header=0, engine='python')
train.columns = names

for name in names:
    col = pd.Categorical(train[name])
    train[name] = col.codes

X = train.drop('salaryLevel', axis=1)
y = train['salaryLevel']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)

print('test set accuracy: ')
print(accuracy_score(y_test, y_predict))

test = pd.read_csv('./adult.test', sep=',', quotechar='"', header=0, engine='python')
test.columns = names

for name in names:
    col = pd.Categorical(test[name])
    test[name] = col.codes

X = test.drop('salaryLevel', axis=1)
y = test['salaryLevel']

X_train, X_test, y_train, y_test = train_test_split(X, y)
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)


print('train set accuracy')
print(accuracy_score(y_test, y_predict))