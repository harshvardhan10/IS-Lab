from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

can_data=datasets.load_breast_cancer()

##print(can_data)
##print(can_data{'target'})

X_train,X_test,y_train,y_test=train_test_split(can_data.data, can_data.target,test_size=0.3,random_state=0)



fr=svm.SVC(kernel="linear")
fr.fit(X_train,y_train)
pred=fr.predict(X_test)
print("Accurate:", metrics.accuracy_score(y_test,y_pred=pred))
print("Precision:", metrics.precision_score(y_test,y_pred=pred))
print("Recall:", metrics.recall_score(y_test,y_pred=pred))

print(metrics.classification_report(y_test,y_pred=pred))
