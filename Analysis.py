import MultiLayerPerceptron as MLP
import pandas as pd

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
MLP.MLP_evaluate(X, Y, Input_shape)