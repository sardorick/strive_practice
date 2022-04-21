### Models
from data_handler import x, y, x_test, transformer_pipeline

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
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier



def models_pipe():
    tree_classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Extra Trees": ExtraTreesClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0),
    "AdaBoost": AdaBoostClassifier(random_state=0),
    "Skl GBM": GradientBoostingClassifier(random_state=0),
    "Skl HistGBM": HistGradientBoostingClassifier(random_state=0),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(random_state=0),
    "Catboost": CatBoostClassifier(random_state=0)
    }

    tree_classifiers = {name: pipeline.make_pipeline(transformer_pipeline(), model) for name, model in tree_classifiers.items()}
    return tree_classifiers

# test models to see which one of them has the most accuracy

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)


results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

def predict_models():
    global results
    for model_name, model in models_pipe().items():
        start_time = time.time()
        
        model.fit(x_train, y_train)
        pred = model.predict(x_val)  
        total_time = time.time() - start_time

        results = results.append({"Model":    model_name,
                                "Accuracy": metrics.accuracy_score(y_val, pred)*100,
                                "Bal Acc.": metrics.balanced_accuracy_score(y_val, pred)*100,
                                "Time":     total_time},
                                ignore_index=True)                        
                                
    results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
    results_ord.index += 1
    return results


def cross_vall():
    skf = model_selection.StratifiedKFold(shuffle=True, random_state=0)

    global results
    for model_name, model in models_pipe().items():
        start_time = time.time()
            
        pred = model_selection.cross_val_predict(model, x, y, cv=skf)
        total_time = time.time() - start_time
        results = results.append({"Model":    model_name,
                                "Accuracy": metrics.accuracy_score(y, pred)*100,
                                "Bal Acc.": metrics.balanced_accuracy_score(y, pred)*100,
                                "Time":     total_time},
                                ignore_index=True)

    results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
    results_ord.index += 1 
    return results

def predictor():

    best_model = models_pipe().get("Skl GBM")

    best_model.fit(x_train, y_train)

    test_pred = best_model.predict(x_test)
    return test_pred

sub = pd.DataFrame(predictor(), index=x_test.index, columns=["Survived"])
print(sub.head())