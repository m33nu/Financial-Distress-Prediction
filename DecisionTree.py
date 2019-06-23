import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier

def Accuracy_DT(split):
    split_size = split

    filename = "FinancialData.xlsx"
    df = pd.read_excel(filename,'DataSet2',header= 0,usecols = "B:T")
    X = df.values
    dm = pd.read_excel(filename,'DataSet2',header= 0,usecols = "U")
    Y = dm.values

    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=split_size, random_state = 0)
    clf = DecisionTreeClassifier(criterion = "entropy", max_depth=4, random_state = 0)
    clf.fit(X_train, Y_train)
    confidence = clf.score(X_test, Y_test)
    y_pred = clf.predict(X_test)
    f1score = f1_score(Y_test, y_pred)
    print('accuracy:',confidence)
    report = classification_report(Y_test, y_pred)
    #print("Decision Tree Classifier ", split)
    print(report)
    return confidence, f1score
#print(Accuracy_DT(0.2))