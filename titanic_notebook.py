# Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Extract
df = pd.read_csv(r'~\projects\Titanic\titanic\train.csv')

# Feature engineer

# title passengers


def collect_title(name):
    initial_index = name.find(', ')
    initial_index += 2
    final_index = name.find('.')
    return name[initial_index:final_index]


df['Title'] = df['Name'].apply(collect_title)

# Preprocessing

# data wrangling
# x and y
features = ['Pclass', 'Title', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']
x = df[features].copy()
y = df['Survived'].copy()

# Treatment of missing values
x['Embarked'].fillna(x['Embarked'].mode().values[0], inplace=True)
x['Age'].fillna(x['Age'].mean(), inplace = True)
# split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

# Baseline
index_female = x_train[x_train.Sex == 'female'].index
sum_female = y_train[index_female].sum()

index_male = x_train[x_train.Sex == 'male'].index
sum_male = y_train[index_male].sum()

sum_female / x_train[x_train.Sex == 'female'].shape[0]
sum_male / x_train[x_train.Sex == 'male'].shape[0]

# my heuristic... if woman... survive!
def survive_if_woman(x_train):
    if x_train.Sex == 'female':
        return 1
    return 0

predictions = x_train.apply(survive_if_woman, axis=1)
f1_score(y_train, predictions)

# Target encoder
category_features = ['Title', 'Sex', 'Ticket', 'Embarked']
num_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

encoder = TargetEncoder(cols=category_features)
encoder.fit(x_train, y_train)

x_train = encoder.transform(x_train)
x_test = encoder.transform(x_test)

# Standard Scaler
scaler_x = StandardScaler()
scaler_x.fit(x_train)

x_train = scaler_x.transform(x_train)
x_test = scaler_x.transform(x_test)

#  Naive Bayes
model_naive = GaussianNB()
model_naive.fit(x_train, y_train)
predict_naive = model_naive.predict(x_test)
print(classification_report(y_test, predict_naive))

# Random Forest
# baseline randomforest
model = RandomForestClassifier(random_state=8)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
f1_score(y_test, predictions)

# KNN
knn_fit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
knn_fit.fit(x_train, y_train)
knn_predictions = knn_fit.predict(x_test)

# Logistc Regression
logistc_fit = LogisticRegression(random_state=1)
logistc_fit.fit(x_train, y_train)
logistc_prediction = logistc_fit.predict(x_test)
