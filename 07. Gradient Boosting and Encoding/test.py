from catboost import train
from train import train
from train import x_test, x_train, y_train, y_test
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

hyper_params = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.001, 0.01, 0.1, 1]
}

GS = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid = hyper_params, scoring = ["r2", "neg_root_mean_squared_error"],
                    refit = "r2", cv = 5, verbose=4)

GS.fit(x_train, y_train)

print(GS.best_params_)

# best params: 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500