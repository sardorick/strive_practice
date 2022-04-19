from sklearn.model_selection import cross_val_score
import data_handler as dh
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor




models = [RandomForestRegressor(), XGBRegressor(), AdaBoostRegressor(), GradientBoostingRegressor(learning_rate=0.01, max_depth=3, n_estimators=500)] # took the params from the GridSearchCV on test.py

x_train, x_test, y_train, y_test = dh.get_data(r"C:\Users\Lenovo\OneDrive\Documents\Strive repos\strive_practice\07. Gradient Boosting and Encoding\insurance.csv")


def train(x_train, x_test, y_train, y_test):
    scores = []
    for model in models:
        model.fit(x_train, y_train)
        # check score
        score = model.score(x_test, y_test)
        scores.append(score)
    return scores

print(train(x_train, x_test, y_train, y_test))

