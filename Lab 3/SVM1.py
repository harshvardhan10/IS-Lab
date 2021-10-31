from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pandas as pd

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = pd.Series(cancer.target)
# print(df.head())

# Dropping fields related to fractal dimension to reduce degree of freedom
df.drop(["mean fractal dimension", "fractal dimension error", "worst fractal dimension"], axis=1, inplace=True)
# print(df.head())

data = df.drop(['target'], axis=1).copy()
# print(data.head())
target = df['target']
# print(target.head())
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
clf = svm.SVC(kernel='linear')  # Linear Kernel

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", metrics.recall_score(y_test, y_pred))
