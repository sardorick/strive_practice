### Data handler

import time
from IPython.display import clear_output
import numpy    as np
import pandas   as pd
import seaborn  as sb
import matplotlib.pyplot as plt
import sklearn  as skl

from sklearn.pipeline import Pipeline      
from sklearn import pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn import compose
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, plot_confusion_matrix         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import set_config
from sklearn.compose import ColumnTransformer


set_config(display='diagram') # Useful for display the pipeline

# read csv files and assign them to pandas dataframes
df = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Documents\Strive repos\strive_practice\08. Robust ML\data\train.csv", index_col='PassengerId')
df_test = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Documents\Strive repos\strive_practice\08. Robust ML\data\test.csv",  index_col='PassengerId')

# make a lambda function to extract titles from full names and apply it
get_Title_from_Name = lambda x: x.split(',')[1].split('.')[0].strip()
df['Title'] = df['Name'].apply(get_Title_from_Name)
df_test['Title'] = df_test['Name'].apply(get_Title_from_Name)

title_dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

# map function

def title_map(df):
    df["Title"] = df["Title"].map(title_dictionary) 
    return df

# Global variables - excluded them from a function
x = title_map(df).drop(columns=["Survived", 'Name', 'Ticket', 'Cabin']) # X DATA (WILL BE TRAIN+VALID DATA)
y = title_map(df)["Survived"] # 0 = No, 1 = Yes

x_test = title_map(df_test).drop(columns=['Name', 'Ticket', 'Cabin']) # # X_TEST DATA (NEW DATA)

cat_vars  = ['Sex', 'Embarked', 'Title']         # x.select_dtypes(include=[object]).columns.values.tolist()
num_vars  = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age'] # x.select_dtypes(exclude=[object]).columns.values.tolist()


# transformer pipeline functio

def transformer_pipeline():
    num_4_treeModels = Pipeline( [ ('imputer', SimpleImputer()) ] )
    cat_4_treeModels = Pipeline( [ ('imputer', SimpleImputer(strategy='most_frequent')), ('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1))])
    tree_prepro = ColumnTransformer( [ ('num', num_4_treeModels, num_vars), ('cat', cat_4_treeModels, cat_vars) ], remainder='drop' )
    return tree_prepro

