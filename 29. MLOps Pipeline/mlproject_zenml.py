import numpy as np
import pandas as pd
from  zenml import step, pipeline

from sklearn.model_selection import train_test_split
from typing import Tuple
import mlflow
from zenml.client import Client
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from zenml.integrations.mlflow.steps.mlflow_registry import (
    mlflow_register_model_step,
)


"""age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
"""



@step
def load_data(url : str) -> pd.DataFrame:
    """Load data from URL"""
    #https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    col = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
               "race", "sex", "capital-gain", "capital-loss" ,"hours-per-week" ,"native-country" , "salary"]
    return pd.read_csv(url, names=col )

@step
def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data"""
    data = data.dropna()
    data = data.drop_duplicates()
    data['salary'] = data['salary'].replace({' >50K':1,' <=50K':0})
    categorical_column_names = data.select_dtypes(exclude = np.number).columns.tolist()
    data = pd.get_dummies(data, columns=categorical_column_names)
    data = data.replace({True:1, False:0})
    return data

@step
def splitting_train_test(data: pd.DataFrame) -> (Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]):
    """Split data into train and test"""
    X = data.drop('salary', axis=1)
    y = data['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(X_train)
    return (X_train, X_test, y_train, y_test)


# # Register the MLflow model registry
# zenml model-registry register mlflow_salary_prediction_v2 --flavor=mlflow

# # Update our stack to include the model registry
# zenml stack update salary_prediction -r mlflow_salary_prediction_v2

experiment_tracker = Client().active_stack.experiment_tracker
print(experiment_tracker.name)

@step(experiment_tracker=experiment_tracker.name)
def randomforest_model_mlflow(X_train : pd.DataFrame, y_train : pd.Series) -> ClassifierMixin:
    """Train a random forest model"""
    mlflow.sklearn.autolog()
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    print('Train Model Accuracy = ', train_acc)
    return model


@step(experiment_tracker=experiment_tracker.name)
def gbm_model_mlflow(X_train : pd.DataFrame, y_train : pd.Series) -> ClassifierMixin:
    """Train a random forest model"""
    mlflow.sklearn.autolog()
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    print('Train Model Accuracy = ', train_acc)
    from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
    print("-------------------",get_tracking_uri())
    return model


@step
def best_model_selector(
    X_test : pd.DataFrame,
    y_test : pd.Series,
    randomforest_model_mlflow : ClassifierMixin,
    gbm_model_mlflow : ClassifierMixin
) -> ClassifierMixin:
    """Select the best model"""
    rf_acc = randomforest_model_mlflow.score(X_test, y_test)
    gbm_acc = gbm_model_mlflow.score(X_test, y_test)
    if rf_acc > gbm_acc:
        return randomforest_model_mlflow
    else:
        return gbm_model_mlflow
    


# most_recent_model_version_number = int(
#     Client()
#     .active_stack.model_registry.list_model_versions(metadata={})[0]
#     .version
# )
# print("most_recent_model_version_number : ",most_recent_model_version_number)

from zenml.integrations.mlflow.steps.mlflow_registry import (
    mlflow_register_model_step,
)

model_name = "zenml-quickstart-model"

register_model = mlflow_register_model_step.with_options(
    parameters=dict(
        name=model_name,
        description="The first run of the Quickstart pipeline.",
    )
)


@pipeline
def train_register_best_model() -> None:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    d = load_data(url)
    d = preprocessing(d)
    X_train, X_test, y_train, y_test = splitting_train_test(d)
    rf_model = randomforest_model_mlflow(X_train, y_train)
    gbm_model = gbm_model_mlflow(X_train, y_train)
    best_model = best_model_selector(X_test, y_test, rf_model, gbm_model)
    register_model(best_model)




if __name__ == "__main__":

    train_register_best_model()



# zenml clean -y
# zenml down
# zenml init
# run mlproject_zenml.py
# zenml up
    





"""url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
d = load_data(url)
print(d.head())
d = preprocessing(d)
print(d.head())
print(d.columns)
X_train, X_test, y_train, y_test = splitting_train_test(d)
print(y_train)"""