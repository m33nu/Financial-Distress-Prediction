import pandas as pd
import NeuralNetwork as NN
import SVM as svm
import DecisionTree as DT
import matplotlib.pyplot as plt

print("\033[32m\nFDP using Neural Network Classifier(60:40):\n" )
NN_a4, NN_f4 = NN.Accuracy_NN(0.4)
print("\033[34mFDP using Support Vector Machines Classifier(60:40):\n" )
svm_a4, svm_f4 = svm.Accuracy_SVM(0.4)
print("\033[35mFDP using Decision Tree Classifier(60:40):\n" )
DT_a4,DT_f4 = DT.Accuracy_DT(0.4)

print("\033[32m\nFDP using Neural Network Classifier(70:30):\n" )
NN_a3, NN_f3 = NN.Accuracy_NN(0.3)
print("\033[34mFDP using Support Vector Machines Classifier(70:30):\n")
svm_a3, svm_f3 = svm.Accuracy_SVM(0.3)
print("\033[35mFDP using Decision Tree Classifier(70:30):\n" )
DT_a3, DT_f3 = DT.Accuracy_DT(0.3)

print("\033[32m\nFDP using Neural Network Classifier(80:20):\n" )
NN_a2, NN_f2 = NN.Accuracy_NN(0.2)
print("\033[34mFDP using Support Vector Machines Classifier(80:20):\n" )
svm_a2, svm_f2 = svm.Accuracy_SVM(0.2)
print("\033[35mFDP using Decision Tree Classifier(80:20):\n" )
DT_a2, DT_f2 = DT.Accuracy_DT(0.2)

models = ("NN", "DT", "SVM")
Partitions =("60:40","70:30","80:20")
NN_results = (NN_a4,NN_a3,NN_a2)
SVM_results = (svm_a4,svm_a3,svm_a2)
DT_results = (DT_a4,DT_a3,DT_a2)

fig, ax = plt.subplots()
ax.plot(Partitions, NN_results, color='r', marker='o', linestyle='--', markersize=5, label='NN')
ax.plot(Partitions, DT_results, color='b', marker='o', linestyle='--', markersize=5, label="DT")
ax.plot(Partitions, SVM_results, color='g', marker='o',linestyle='--', markersize=5, label="SVM")

legend = ax.legend(loc='lower right', shadow=True)

data = [[format(NN_a4, '.2f'), format(NN_a3, '.2f'), format(NN_a2,'.2f')], [format(DT_a4, '.2f'), format(DT_a3, '.2f'), format(DT_a2,'.2f')], [format(svm_a4, '.2f'), format(svm_a3, '.2f'), format(svm_a2, '.2f')]]
print("\033[33m\nAccuracy Table(Classification\Train:Test Ratio)\n", pd.DataFrame(data, models, Partitions))

plt.ylim([0.2,1])
plt.xlabel("Data Partition ratios")
plt.ylabel("Percentage Accuracy")
plt.title("Accuracy of Machine Learning Classifiers on FDP")
plt.show()

