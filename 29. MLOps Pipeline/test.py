import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
col = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
               "race", "sex", "capital-gain", "capital-loss" ,"hours-per-week" ,"native-country" , "salary"]
data = pd.read_csv(url, names=col, na_values='?')

print(data.isna().sum())

print(data['salary'].value_counts())

data['salary'] = data['salary'].replace({' >50K':1,' <=50K':0})
print(data['salary'].value_counts())
