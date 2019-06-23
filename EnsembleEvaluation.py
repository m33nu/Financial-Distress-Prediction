import MajorityVoting as MV
import RandomForest as RF
import Adaboost as AB
import matplotlib.pyplot as plt1
import pandas as pd

print("\033[32m\nFDP using Majority Voting(60:40):\n" )
mv_a4,mv_f4 = MV.Accuracy_MV(0.4)
print("\033[32m\nFDP using Random Forest(60:40):\n" )
rf_a4, rf_f4 = RF.Accuracy_RF(0.4)
print("\033[32m\nFDP using AdaBoost(60:40):\n" )
ab_a4, ab_f4 = AB.Accuracy_AB(0.4)

print("\033[32m\nFDP using Majority Voting(70:30):\n" )
mv_a3, mv_f3 = MV.Accuracy_MV(0.3)
print("\033[32m\nFDP using Random Forest(70:30):\n" )
rf_a3, rf_f3 = RF.Accuracy_RF(0.3)
print("\033[32m\nFDP using AdaBoost(70:30):\n" )
ab_a3, ab_f3 = AB.Accuracy_AB(0.3)

print("\033[32m\nFDP using Majority Voting(80:20):\n" )
mv_a2, mv_f2 = MV.Accuracy_MV(0.2)
print("\033[32m\nFDP using Random Forest(80:20):\n" )
rf_a2, rf_f2 = RF.Accuracy_RF(0.2)
print("\033[32m\nFDP using AdaBoost(80:20):\n" )
ab_a2, ab_f2 = AB.Accuracy_AB(0.2)

models = ("Majority Voting","Random Forest", "AdaBoost")
Partitions =("60:40","70:30","80:20")
MV_results = (mv_a4, mv_a3, mv_a2);
RF_results = (rf_a4, rf_a3, rf_a2);
AB_results = (ab_a4, ab_a3, ab_a2);
MV_F1results = (mv_f4, mv_f3, mv_f2);
RF_F1results = (rf_f4, rf_f3, rf_f2);
AB_F1results = (ab_f4, ab_f3, ab_f2);


fig, ax = plt1.subplots()
plt1.title('Accuracy of Ensemble Techniques on FDP')
ax.plot(Partitions, MV_results, color='r', marker='o', linestyle='--', markersize=5, label='Majority Voting')
ax.plot(Partitions, RF_results, color='b', marker='o', linestyle='--', markersize=5, label="RandomForest")
ax.plot(Partitions, AB_results, color='g', marker='o',linestyle='--', markersize=5, label="AdaBoost")


legend = ax.legend(loc='lower right', shadow=True)

data = [[round(mv_a4,2), round(mv_a3,2), round(mv_a2,2)], [round(rf_a4,2), round(rf_a3,2), round(rf_a2,2)], [round(ab_a4,2), round(ab_a3,2), round(ab_a2,2)]]
print("\033[33m\nAccuracy Table(Ensemble Methods\Train:Test Ratio)\n", pd.DataFrame(data, models, Partitions))


plt1.ylim([0.2,1])
plt1.xlabel("Data Partition ratios")
plt1.ylabel("Percentage Accuracy")
plt1.show()
