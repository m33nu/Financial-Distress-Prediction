import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score,accuracy_score

def Accuracy_NN(split):
    split_size = split

    filename = "FinancialData.xlsx"
    df = pd.read_excel(filename,'DataSet2',header= 0,usecols = "B:T")
    X = df.values
    dm = pd.read_excel(filename,'DataSet2',header= 0,usecols = "U")
    Y = dm.values

    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=split_size, random_state = 60)
    clf = MLPClassifier(hidden_layer_sizes=(10,10), random_state = 100)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    f1score = f1_score(Y_test, y_pred)
    accuracy = accuracy_score(Y_test, y_pred)
    print('accuracy:',accuracy)
    report = classification_report(Y_test, y_pred)
    print(report)
    return accuracy, f1score
#print(Accuracy_NN(0.4))