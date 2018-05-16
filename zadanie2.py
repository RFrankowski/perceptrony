import pandas as pd
from nn import  NeuralNetwork
import numpy as np

# import statsmodels as sm
# import sklearn as skl
# import sklearn.preprocessing as preprocessing
# import sklearn.linear_model as linear_model
# import sklearn.cross_validation as cross_validation
# import sklearn.metrics as metrics
# import sklearn.tree as tree
# import seaborn as sns

# age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, income

original_data = pd.read_csv(
    "adult.data",
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
    sep=r'\s*,\s*',
    engine='python',
    na_values="?")

# print original_data.tail()
raw_data = original_data[["Age", "Hours per week", "Sex"]].head(15)
train_data = []
for i in enumerate(raw_data["Age"]):
    train_data.append([[raw_data["Age"][i[0]]], [raw_data["Hours per week"][i[0]]]])

train_data = np.array(train_data)
print train_data

nn = NeuralNetwork(2, 2, 1)


def ucz(per, tr_data):
    for k in range(100):
        for tr in tr_data:
            per.train(tr, np.array([0]))
            print " tutaj powinno byc   a jest " + str(per.predict(tr))
ucz(nn,train_data)