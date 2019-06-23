import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import SVC

def Accuracy_SVM(split):
    split_size = split

    filename = "FinancialData.xlsx"
    df = pd.read_excel(filename,'DataSet2',header= 0,usecols = "B:T")
    X = df.values
    dm = pd.read_excel(filename,'DataSet2',header= 0,usecols = "U")
    Y = dm.values

    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=split_size, random_state = 100)
    clf = SVC(kernel='sigmoid', random_state = 100)
    clf.fit(X_train, Y_train)
    confidence = clf.score(X_test, Y_test)
    y_pred = clf.predict(X_test)
    f1score = f1_score(Y_test, y_pred)
    print('accuracy:',confidence)
    report = classification_report(Y_test, y_pred)
    #print("SVM Classifier ", split)
    print(report)
    return confidence, f1score
#print(Accuracy_SVM(0.3))