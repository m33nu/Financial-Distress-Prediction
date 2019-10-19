import pandas as pd
import Classifiers as NN
import matplotlib.pyplot as plt

def get_dataset1():
    """Retrieve the kuwait dataset and process the data."""
    # Set defaults.
    filename = "Kuwait.xlsx"
    df = pd.read_excel(filename, 'DataSet2', header= 0, usecols="B:X")
    dm = pd.read_excel(filename, 'DataSet2', header= 0, usecols="AF")
    X = df.values
    Y = dm.values
    input_shape = 23
    return X, Y, input_shape


def get_dataset2():
    """Retrieve the gcc dataset and process the data."""
    # Set defaults.
    filename = "FinancialData.xlsx"
    df = pd.read_excel(filename, 'DataSet2', header=0, usecols="B:R")
    dm = pd.read_excel(filename, 'DataSet2', header=0, usecols="Y")
    X = df.values
    Y = dm.values
    input_shape = 17
    return X, Y, input_shape


X, Y, Input_shape = get_dataset1()
D1_MV_accuracy, D1_MV_f1score = NN.Accuracy_MV(X, Y)
print("MV Dataset1: ", D1_MV_accuracy, D1_MV_f1score)
D1_RF_accuracy, D1_RF_f1score = NN.Accuracy_RF(X, Y)
print("RF Dataset1: ", D1_RF_accuracy, D1_RF_f1score)
D1_ADA_accuracy, D1_ADA_f1score = NN.Accuracy_Adaboost(X, Y)
print("Adaboost Dataset1: ", D1_ADA_accuracy, D1_ADA_f1score)
D1_DT_accuracy, D1_DT_f1score = NN.Accuracy_DT(X, Y)
print("DT Dataset1: ", D1_DT_accuracy, D1_DT_f1score)
D1_SVM_accuracy, D1_SVM_f1score = NN.Accuracy_SVM(X, Y)
print("SVM Dataset1: ", D1_SVM_accuracy, D1_SVM_f1score)
D1_NN_accuracy, D1_NN_f1score = (NN.Accuracy_NN(X, Y))
print("NN Dataset1: ", D1_NN_accuracy, D1_NN_f1score)

X, Y, Input_shape = get_dataset2()
D2_MV_accuracy, D2_MV_f1score = NN.Accuracy_MV(X, Y)
print("MV Dataset2: ", D2_MV_accuracy, D2_MV_f1score)
D2_RF_accuracy, D2_RF_f1score = NN.Accuracy_RF(X, Y)
print("RF Dataset2: ", D2_RF_accuracy, D2_RF_f1score)
D2_ADA_accuracy, D2_ADA_f1score = NN.Accuracy_Adaboost(X, Y)
print("Adaboost Dataset2: ", D2_ADA_accuracy, D2_ADA_f1score)
D2_DT_accuracy, D2_DT_f1score = NN.Accuracy_DT(X, Y)
print("DT Dataset2: ", D2_DT_accuracy, D2_DT_f1score)
D2_SVM_accuracy, D2_SVM_f1score = NN.Accuracy_SVM(X, Y)
print("SVM Dataset2: ", D2_SVM_accuracy, D2_SVM_f1score)
D2_NN_accuracy, D2_NN_f1score = NN.Accuracy_NN(X, Y)
print("NN Dataset2: ", D2_NN_accuracy, D2_NN_f1score)

dataset = ("Dataset1", "Dataset2")
models = ("NN", "DT", "SVM", "MV", "RF", "Adaboost")
Dataset1_Accuracy = (D1_NN_accuracy, D1_DT_accuracy, D1_SVM_accuracy, D1_MV_accuracy, D1_RF_accuracy, D1_ADA_accuracy)
Dataset2_Accuracy = (D2_NN_accuracy, D2_DT_accuracy, D2_SVM_accuracy, D2_MV_accuracy, D2_RF_accuracy, D2_ADA_accuracy)
Dataset1_f1score = (D1_NN_f1score, D1_DT_f1score, D1_SVM_f1score, D1_MV_f1score, D1_RF_f1score, D1_ADA_f1score)
Dataset2_f1score = (D2_NN_f1score, D2_DT_f1score, D2_SVM_f1score, D2_MV_f1score, D2_RF_f1score, D2_ADA_f1score)


fig, ax = plt.subplots()
ax.plot(models, Dataset1_Accuracy, color='r', marker='o', linestyle='--', markersize=5, label='Dataset1')
ax.plot(models, Dataset2_Accuracy, color='b', marker='o', linestyle='--', markersize=5, label="DataSet2")

legend = ax.legend(loc='lower right', shadow=True)

data = [[format(D1_NN_accuracy, '.2f'), format(D2_NN_accuracy, '.2f')], [format(D1_DT_accuracy, '.2f'), format(D2_DT_accuracy, '.2f')], [format(D1_SVM_accuracy, '.2f'), format(D2_SVM_accuracy, '.2f')], [format(D1_MV_accuracy, '.2f'), format(D2_MV_accuracy, '.2f')],[format(D1_RF_accuracy, '.2f'), format(D2_RF_accuracy, '.2f')], [format(D1_ADA_accuracy, '.2f'), format(D2_ADA_accuracy, '.2f')]]
print("\033[33m\nAccuracy Table(Classification\Train:Test Ratio)\n", pd.DataFrame(data, models, dataset))

plt.ylim([20,100])
plt.xlabel("Classifier Model")
plt.ylabel("Mean Accuracy")
plt.title("Accuracy of Machine Learning Classifiers on FDP")
plt.show()

fig, ax = plt.subplots()
ax.plot(models, Dataset1_f1score, color='r', marker='o', linestyle='--', markersize=5, label='Dataset1')
ax.plot(models, Dataset2_f1score, color='b', marker='o', linestyle='--', markersize=5, label="DataSet2")

legend = ax.legend(loc='lower right', shadow=True)

data = [[format(D1_NN_f1score, '.2f'), format(D2_NN_f1score, '.2f')], [format(D1_DT_f1score, '.2f'), format(D2_DT_f1score, '.2f')], [format(D1_SVM_f1score, '.2f'), format(D2_SVM_f1score, '.2f')], [format(D1_MV_f1score, '.2f'), format(D2_MV_f1score, '.2f')],[format(D1_RF_f1score, '.2f'), format(D2_RF_f1score, '.2f')], [format(D1_ADA_f1score, '.2f'), format(D2_ADA_f1score, '.2f')]]
print("\033[33m\nAccuracy Table(Classification\Train:Test Ratio)\n", pd.DataFrame(data, models, dataset))

plt.ylim([20,100])
plt.xlabel("Classifier Model")
plt.ylabel("Mean F1-score")
plt.title("F1-Score of Machine Learning Classifiers on FDP")
plt.show()

