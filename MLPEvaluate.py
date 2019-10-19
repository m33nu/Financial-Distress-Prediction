import MultiLayerPerceptron as MLP
import pandas as pd
import matplotlib.pyplot as plt1

def get_dataset1():
    """Retrieve the kuwait dataset and process the data."""
    # Set defaults.
    global input_shape
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
    global input_shape
    filename = "FinancialData.xlsx"
    df = pd.read_excel(filename, 'DataSet2', header=0, usecols="B:R")
    dm = pd.read_excel(filename, 'DataSet2', header=0, usecols="Y")
    X = df.values
    Y = dm.values
    input_shape = 17
    return X, Y, input_shape


X, Y, Input_shape = get_dataset1()
Layers = ("1","2","3")
MLPresult1, MLPresult2, MLPresult3,MLPf1, MLPf2, MLPf3 = MLP.MLP_evaluate(X, Y, Input_shape)
MLResults = (MLPresult1,MLPresult2,MLPresult3)
MLf1Results = (MLPf1, MLPf2, MLPf3)

X, Y, Input_shape = get_dataset2()
MLPresult1, MLPresult2, MLPresult3,MLPf1, MLPf2, MLPf3 = MLP.MLP_evaluate(X, Y, Input_shape)

D2_MLResults = (MLPresult1,MLPresult2,MLPresult3)
D2_MLf1Results = (MLPf1, MLPf2, MLPf3)


fig, ax = plt1.subplots()
plt1.title('F1-score of Deep Neural Network(MLP) on FDP')
ax.plot(Layers, MLf1Results, color='r', marker='o', linestyle='--', markersize=5, label='Dataset1')
ax.plot(Layers, D2_MLf1Results, color='b', marker='o', linestyle='--', markersize=5, label='Dataset2')
legend = ax.legend(loc='lower right', shadow=True)

plt1.ylim([0,100])
plt1.xlabel("No: of Layers")
plt1.ylabel("Kfold Mean f1-score")

plt1.show()

fig1, ax1 = plt1.subplots()
plt1.title('Accuracy of Deep Neural Network(MLP) on FDP')
ax1.plot(Layers, MLResults, color='r', marker='o', linestyle='--', markersize=5, label='Dataset1')
ax1.plot(Layers, D2_MLResults, color='b', marker='o', linestyle='--', markersize=5, label='Dataset2')


legend = ax1.legend(loc='lower right', shadow=True)

plt1.ylim([0,100])
plt1.xlabel("No: of Layers")
plt1.ylabel("Kfold Mean Accuracy")


plt1.show()