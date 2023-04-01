import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data=pd.read_csv(r"/home/prem/Desktop/pythonpersonal/my_project/diabetes.csv")
data.head()
data.info()
data.isnull().sum()
a=data['Outcome'].value_counts()[0]
b=data['Outcome'].value_counts()[1]
# print("Percentage of non-diabetic patient",(a/(a+b))*100)
# print("Percentage of diabetic patient",(b/(a+b))*100)

data['Pregnancies']=data['Pregnancies'].clip(upper=data['Pregnancies'].quantile(0.77))
data['Age']=data['Age'].clip(upper=data['Age'].quantile(0.76))
data['Glucose']=data['Glucose'].clip(lower=data['Glucose'].quantile(0.20))
data['BloodPressure']=data['BloodPressure'].clip(lower=data['BloodPressure'].quantile(0.24),upper=data['BloodPressure'].quantile(0.76))
data['SkinThickness']=data['SkinThickness'].clip(upper=data['SkinThickness'].quantile(0.80))
data['Insulin']=data['Insulin'].clip(upper=data['Insulin'].quantile(0.75))
data['BMI']=data['BMI'].clip(lower=data['BMI'].quantile(0.23),upper=data['BMI'].quantile(0.77))
data['DiabetesPedigreeFunction']=data['DiabetesPedigreeFunction'].clip(upper=data['DiabetesPedigreeFunction'].quantile(0.75))
corr=data.corr()
data.groupby('Outcome').hist(figsize=(14, 13))

label_encoder = preprocessing.LabelEncoder()
data['Pregnancies']=label_encoder.fit_transform(data['Pregnancies'])
data['Age']= label_encoder.fit_transform(data['Age'])
data['Glucose']= label_encoder.fit_transform(data['Glucose'])
data['BloodPressure']= label_encoder.fit_transform(data['BloodPressure'])
data['SkinThickness']= label_encoder.fit_transform(data['SkinThickness'])
data['Insulin']= label_encoder.fit_transform(data['Insulin'])
data['BMI']= label_encoder.fit_transform(data['BMI'])
data['DiabetesPedigreeFunction']= label_encoder.fit_transform(data['DiabetesPedigreeFunction'])


outcome_count_0, outcome_count_1 = data['Outcome'].value_counts()
outcome_0 = data[data['Outcome'] == 0]
outcome_1 = data[data['Outcome'] == 1]
print('Outcome 0:', outcome_0.shape)
print('Outcome 1:', outcome_1.shape)

outcome_0_under = outcome_0.sample(outcome_count_1)
test_under = pd.concat([outcome_0_under, outcome_1], axis=0)
# print("Total Outcomes of 1 and 0:",test_under['Outcome'].value_counts())
test_under['Outcome'].value_counts().plot(kind='bar', title='count (target)')


outcome_1_over = outcome_1.sample(outcome_count_0, replace=True)
test_over = pd.concat([outcome_1_over, outcome_0], axis=0)
# print("total class of 1 and 0:",test_over['Outcome'].value_counts())
test_over['Outcome'].value_counts().plot(kind='bar', title='count (target)')

X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25,random_state=42)

def build_random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred1 = rf.predict(X_test)
    acc1 = accuracy_score(y_test, y_pred1)   
    return acc1
acc1 = build_random_forest(X_train, y_train, X_test, y_test)
print("Accuracy score: {:.5f}".format(acc1))
acc1=acc1*100
print(acc1)


def build_xgb_classifier(X_train, y_train, X_test,y_test):
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.fit(X_train, y_train)
    y_pred2 = xgb_clf.predict(X_test)
    acc2 = accuracy_score(y_test, y_pred2)
    return acc2
acc2 = build_xgb_classifier(X_train, y_train, X_test, y_test)
print("Accuracy score: {:.5f}".format(acc2))
acc2=acc2*100
print(acc2)


def build_naive_bayes_classifier(X_train, y_train, X_test,y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred3 = gnb.predict(X_test)
    acc3 = accuracy_score(y_test, y_pred3)
    return acc3
acc3 = build_naive_bayes_classifier(X_train, y_train, X_test, y_test)
print("Accuracy score: {:.5f}".format(acc3))
acc3=acc3*100
print(acc3)


def build_decision_tree_classifier(X_train, y_train, X_test, y_test):
    dt = DecisionTreeClassifier(criterion='gini',max_depth=None, min_samples_split=2)
    dt.fit(X_train, y_train)
    y_pred4 = dt.predict(X_test)
    acc4 = accuracy_score(y_test, y_pred4)
    return acc4
acc4 = build_decision_tree_classifier(X_train, y_train, X_test, y_test)
print("Accuracy score: {:.5f}".format(acc4))
acc4=acc4*100
print(acc4)


algorithm_accuracy = {
    'Random Forest Classifier': acc1,
    'XG Boost Classifier': acc2,
    'Naive Bayes Classifier': acc3,
    'Decision Tree Classifier': acc4,
}

df = pd.DataFrame(list(algorithm_accuracy.items()), columns=['ML Algorithm', 'Accuracy'])
print(df)

def prediction_with_random_forest(new_data):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    new_pred = rf_model.predict(new_data)
    return new_pred

# row = pd.DataFrame(np.array([[8,183,64,0,0,23.3,0.672,32]]), columns=X_train.columns)
# new_pred = prediction_with_random_forest(rf_model, row)
# print(new_pred)
