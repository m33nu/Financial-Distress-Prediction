import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

def Accuracy_MV(split):
    split_size = split

    filename = "FinancialData.xlsx"
    df = pd.read_excel(filename,'DataSet2',header= 0,usecols = "B:T")
    X = df.values
    dm = pd.read_excel(filename,'DataSet2',header= 0,usecols = "V")
    Y = dm.values

    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=split_size, random_state = 100)


    clf = VotingClassifier(estimators = [('NN',MLPClassifier(hidden_layer_sizes=(10,10), random_state = 0)),
                                ('svc',SVC(kernel='sigmoid', random_state = 100)),
                                ('DT', DecisionTreeClassifier(criterion = "entropy", random_state = 80, max_depth=4))], voting = "hard")


    clf.fit(X_train, Y_train)
    confidence = clf.score(X_test, Y_test)
    y_pred = clf.predict(X_test)
    print('accuracy:',confidence)
    f1score = f1_score(Y_test,y_pred)
    report = classification_report(Y_test, y_pred)
    #print("Majority Voting ", split)
    print(report)
    return confidence, f1score
#print(Accuracy_MV(0.2))