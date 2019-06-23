import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score


def Accuracy_AB(split):
    split_size = split

    filename = "FinancialData.xlsx"
    df = pd.read_excel(filename,'DataSet2',header= 0,usecols = "B:T")
    X = df.values
    dm = pd.read_excel(filename,'DataSet2',header= 0,usecols = "U")
    Y = dm.values


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, random_state=0)

    svc = SVC(probability=True, kernel='sigmoid')
    adaboostClassifier = AdaBoostClassifier(n_estimators=50, base_estimator=svc, learning_rate=1)

    model = adaboostClassifier.fit(X_train, Y_train)

    y_pred = model.predict(X_test)
    f1score = f1_score(Y_test, y_pred)
    accuracy = accuracy_score(Y_test,y_pred)
    report = classification_report(Y_test, y_pred)
    print('accuracy:', accuracy)
    #print("Adaboost ", split)
    print(report)
    return accuracy, f1score
#print(Accuracy_AB(0.3))