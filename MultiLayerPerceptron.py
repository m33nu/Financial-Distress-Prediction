import tensorflow as tf
import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
input_shape = 0

# baseline model
def deep_model_1layer():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=input_shape, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.train.AdamOptimizer(0.001) , metrics=['accuracy'])
    return model


def deep_model_2layer():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=input_shape, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.train.AdamOptimizer(0.001) , metrics=['accuracy'])
    return model


def deep_model_3layer():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=input_shape, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.train.AdamOptimizer(0.001) , metrics=['accuracy'])
    return model


def deep_model_4layer():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=input_shape, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.train.AdamOptimizer(0.001) , metrics=['accuracy'])
    return model

def deep_model_5layer():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=input_shape, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.train.AdamOptimizer(0.001) , metrics=['accuracy'])
    return model

def MLP_evaluate(X, Y, input_dim):
    global input_shape
    input_shape = input_dim
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=100)
    scoring = {'accuracy': make_scorer(accuracy_score),
               'f1_score': make_scorer(f1_score)}

    estimator1 = KerasClassifier(build_fn=deep_model_1layer, epochs=150, batch_size=12, verbose=0)
    results1 = cross_validate(estimator1, X, Y, cv=kfold, scoring=scoring)
    estimator2 = KerasClassifier(build_fn=deep_model_2layer, epochs=150, batch_size=12, verbose=0)
    results2 = cross_validate(estimator2, X, Y, cv=kfold, scoring=scoring)
    estimator3 = KerasClassifier(build_fn=deep_model_3layer, epochs=150, batch_size=12, verbose=0)
    results3 = cross_validate(estimator3, X, Y, cv=kfold, scoring=scoring)
    estimator4 = KerasClassifier(build_fn=deep_model_4layer, epochs=150, batch_size=12, verbose=0)
    results4 = cross_validate(estimator4, X, Y, cv=kfold, scoring=scoring)
    """estimator5 = KerasClassifier(build_fn=deep_model_5layer, epochs=150, batch_size=12, verbose=0)
    results5 = cross_validate(estimator4, X, Y, cv=kfold, scoring=scoring)
    """
    print("Training 1 Layer: %.2f%% (%.2f%%)" % (results1['train_accuracy'].mean()*100, results1['train_accuracy'].std()*100))
    print("Testing 1 Layer: %.2f%% (%.2f%%)" % (results1['test_accuracy'].mean() * 100, results1['test_accuracy'].std() * 100))
    print("Training 2 Layer: %.2f%% (%.2f%%)" % (results2['train_accuracy'].mean() * 100, results2['train_accuracy'].std() * 100))
    print("Testing 2 Layer: %.2f%% (%.2f%%)" % (results2['test_accuracy'].mean() * 100, results2['test_accuracy'].std() * 100))
    print("Training 3 Layer: %.2f%% (%.2f%%)" % (results3['train_accuracy'].mean() * 100, results3['train_accuracy'].std() * 100))
    print("Testing 3 Layer: %.2f%% (%.2f%%)" % (results3['test_accuracy'].mean() * 100, results3['test_accuracy'].std() * 100))
    print("Training 4 Layer: %.2f%% (%.2f%%)" % (results4['train_accuracy'].mean() * 100, results4['train_accuracy'].std() * 100))
    print("Testing 4 Layer: %.2f%% (%.2f%%)" % (results4['test_accuracy'].mean() * 100, results4['test_accuracy'].std() * 100))
    """
    print("Training 5 Layer: %.2f%% (%.2f%%)" % (results5['train_accuracy'].mean() * 100, results5['train_accuracy'].std() * 100))
    print("Testing 5 Layer: %.2f%% (%.2f%%)" % (results5['test_accuracy'].mean() * 100, results5['test_accuracy'].std() * 100))
    """
    return results1['test_accuracy'].mean() * 100, results2['test_accuracy'].mean() * 100, results3['test_accuracy'].mean() * 100, results1['test_f1_score'].mean() * 100, results2['test_f1_score'].mean() * 100, results3['test_f1_score'].mean() * 100
