import imp
from catboost import train
from train import Main
from train import x_test, x_train, y_train, y_test
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor

models = [RandomForestRegressor(), XGBRegressor(), AdaBoostRegressor(), GradientBoostingRegressor()]

hyper_params_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.001, 0.01, 0.1, 1]
}

hyper_params_rf = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 3, 6, 9],
    "learning_rate": [0.001, 0.01, 0.1, 1],
    "min_samples_leaf": [0.1, 0.01, 1]
}

hyper_params_xgb = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.001, 0.01, 0.1, 1],
    "gamma": [0.01, 0.1]
}

hyper_params_ada = {
    "n_estimators": [100, 200, 500],
    "learning_rate": [0.001, 0.01, 0.1, 1],
    "loss": ['linear', 'square', 'exponential']
}

def grid_searcher(model):
        if model == gradient_booster:
            GS = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid = hyper_params_grid, scoring = ["r2", "neg_root_mean_squared_error"],
                    refit = "r2", cv = 5, verbose=4)
            GS.fit(x_train, y_train)
            print(GS.best_params_)
        elif model == random_booster:
            GS = GridSearchCV(estimator=RandomForestRegressor(), param_grid = hyper_params_rf, scoring = ["r2", "neg_root_mean_squared_error"],
                    refit = "r2", cv = 5, verbose=4)
            GS.fit(x_train, y_train)
            print(GS.best_params_)
        elif model == xg_booster:
            GS = GridSearchCV(estimator=XGBRegressor(), param_grid = hyper_params_xgb, scoring = ["r2", "neg_root_mean_squared_error"],
                    refit = "r2", cv = 5, verbose=4)
            GS.fit(x_train, y_train)
            print(GS.best_params_)
        else:
            GS = GridSearchCV(estimator=AdaBoostRegressor(), param_grid = hyper_params_ada, scoring = ["r2", "neg_root_mean_squared_error"],
                    refit = "r2", cv = 5, verbose=4)
            GS.fit(x_train, y_train)
            print(GS.best_params_)


gradient_booster = GradientBoostingRegressor()
random_booster = RandomForestRegressor()
xg_booster = XGBRegressor()
ada_booster = AdaBoostRegressor()
grid_searcher(gradient_booster)

        

# best params: 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500