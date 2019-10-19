import numpy
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, RandomForestClassifier

seed = 7
numpy.random.seed(seed)

def Accuracy_NN(X,Y):
    model = MLPClassifier(hidden_layer_sizes=(15,15), activation="tanh", random_state=50)
    scoring = {'accuracy': make_scorer(accuracy_score),
               'f1_score': make_scorer(f1_score)}
    results = cross_validate(model, X, Y, cv=10, scoring=scoring)
    print(results)
    return results['test_accuracy'].mean() * 100, results['test_f1_score'].mean() * 100


def Accuracy_SVM(X, Y):

    model = SVC(kernel='poly', random_state = 50)
    scoring = {'accuracy': make_scorer(accuracy_score),
               'f1_score': make_scorer(f1_score)}
    results = cross_validate(model, X, Y, cv=10, scoring = scoring)
    print(results)
    return results['test_accuracy'].mean() * 100,  results['test_f1_score'].mean() * 100


def Accuracy_DT(X, Y):

    model = DecisionTreeClassifier(criterion = "entropy", max_depth=4, random_state = 50)
    scoring = {'accuracy': make_scorer(accuracy_score),
               'f1_score': make_scorer(f1_score)}
    results = cross_validate(model, X, Y, cv=10, scoring = scoring)
    print(results)
    return results['test_accuracy'].mean() * 100,  results['test_f1_score'].mean() * 100


def Accuracy_RF(X,Y):
    model = RandomForestClassifier(n_estimators=20, random_state=0)
    scoring = {'accuracy': make_scorer(accuracy_score),
               'f1_score': make_scorer(f1_score)}
    results = cross_validate(model, X, Y, cv=10, scoring=scoring)
    print(results)
    return results['test_accuracy'].mean() * 100, results['test_f1_score'].mean() * 100


def Accuracy_Adaboost(X, Y):

    svc = SVC(probability=True, kernel='sigmoid')
    model = AdaBoostClassifier(n_estimators=50, base_estimator=svc, learning_rate=1)

    scoring = {'accuracy': make_scorer(accuracy_score),
               'f1_score': make_scorer(f1_score)}
    results = cross_validate(model, X, Y, cv=10, scoring = scoring)
    print(results)
    return results['test_accuracy'].mean() * 100,  results['test_f1_score'].mean() * 100


def Accuracy_MV(X, Y):

    model = VotingClassifier(estimators = [('NN',MLPClassifier(hidden_layer_sizes=(15,15), random_state = 50)),
                                ('svc',SVC(kernel='poly', random_state = 50)),
                                ('DT', DecisionTreeClassifier(criterion = "entropy", random_state = 50, max_depth=4))], voting = "hard")
    scoring = {'accuracy': make_scorer(accuracy_score),
               'f1_score': make_scorer(f1_score)}
    results = cross_validate(model, X, Y, cv=10, scoring = scoring)
    print(results)
    return results['test_accuracy'].mean() * 100,  results['test_f1_score'].mean() * 100
