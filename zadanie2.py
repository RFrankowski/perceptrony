import pandas as pd
from nn2 import NeuralNetwork
import numpy as np

# import statsmodels as sm
# import sklearn as skl
# import sklearn.preprocessing as preprocessing
# import sklearn.linear_model as linear_model
# import sklearn.cross_validation as cross_validation
# import sklearn.metrics as metrics
# import sklearn.tree as tree
# import seaborn as sns

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, income
# male - 1, female - 0
# lessoreq50K 0,more50K 1
# White 0, Asian-Pac-Islander 1, Amer-Indian-Eskimo 2, Other 3 , Black 4.
original_data = pd.read_csv(
    "adult.data",
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Income"],
    sep=r'\s*,\s*',
    engine='python',
    na_values="?")
# , [raw_data["Race"][i[0]]]
# print original_data.tail()
raw_data = original_data[["Age", "Hours per week", "Race", "Sex","Income"]].head(10)
# print raw_data
target_data = []
train_data = []
for i in enumerate(raw_data["Age"]):
    train_data.append([[sigmoid(raw_data["Age"][i[0]])], [raw_data["Sex"][i[0]]], [sigmoid(raw_data["Hours per week"][i[0]])], [raw_data["Income"][i[0]]]])
    if raw_data["Race"][i[0]] == 0:
        target_data.append(np.array([[1], [0], [0], [0], [0]]))
    if raw_data["Race"][i[0]] == 1:
        target_data.append(np.array([[0], [1], [0], [0], [0]]))
    if raw_data["Race"][i[0]] == 2:
        target_data.append(np.array([[0], [0], [1], [0], [0]]))
    if raw_data["Race"][i[0]] == 3:
        target_data.append(np.array([[0], [0], [0], [1], [0]]))
    if raw_data["Race"][i[0]] == 4:
        target_data.append(np.array([[0], [0], [0], [0], [1]]))

target_data = target_data
train_data = np.array(train_data)
print target_data
# print train_data
#
nn = NeuralNetwork(4, 20, 10, 5)


def train(per, tr_data, taget_data):
    for k in range(2000):
        for i in enumerate(tr_data):
            per.train(i[1], taget_data[i[0]])
            print " tutaj powinno byc " + str(taget_data[i[0]]) + " a jest" + str(per.predict(i[1])) + "\n"


train(nn, train_data, target_data)
