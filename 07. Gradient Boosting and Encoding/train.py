from sklearn.model_selection import cross_val_score
import data_handler as dh
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor




models = [RandomForestRegressor(), XGBRegressor(), AdaBoostRegressor(), GradientBoostingRegressor()] # took the params from the GridSearchCV on test.py

x_train, x_test, y_train, y_test = dh.get_data(r"C:\Users\Lenovo\OneDrive\Documents\Strive repos\strive_practice\07. Gradient Boosting and Encoding\insurance.csv")

class Main:
    def __init__(self, x_train, x_test, y_train, y_test, model):
        self.x_train, self.x_test = x_train, x_test
        self.y_train, self.y_test = y_train, y_test
        self.model = model

    def train(self):
        self.fitted = []
        for self.model in models:
            self.fitted.append([self.model.fit(self.x_train, self.y_train)])
            
        return self.fitted


    def scoring_func(self, model):
        self.model = model
        scores = []
        # check score
        for self.fitted in models:
            score = self.fitted.score(self.x_test, self.y_test)
            scores.append(score)
        return scores

test = Main(x_train, x_test, y_train, y_test, models)
print(test.scoring_func(test.train()))

# def train(x_train, x_test, y_train, y_test):
#     for model in models:
#         model.fit(x_train, y_train)
#     return model

# def scoring_func(model):
#     scores = []
#     # check score
#     for model in models:
#         score = model.score(x_test, y_test)
#         scores.append(score)
#     return scores

# print(scoring_func(train(x_train, x_test, y_train, y_test)))

