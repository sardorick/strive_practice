from sklearn.model_selection import cross_val_score
import data_handler as dh
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


# rf_model = RandomForestRegressor()
# ab_model = AdaBoostRegressor(learning_rate=0.1)
# gb_model = GradientBoostingRegressor(learning_rate=0.01)
# xg_model = xgboost.XGBRegressor()

x_train, x_test, y_train, y_test = dh.get_data(r"C:\Users\Lenovo\OneDrive\Documents\Strive repos\strive_practice\07. Gradient Boosting and Encoding\insurance.csv")

# rf_regressor = rf_model.fit(x_train, y_train)
# ab_regressor = ab_model.fit(x_train, y_train)
# gb_regressor = gb_model.fit(x_train, y_train)
# xg_regressor = xg_model.fit(x_train, y_train)

models = [RandomForestRegressor(), GradientBoostingRegressor, XGBRegressor(), AdaBoostRegressor()]

for model in models:
    print(cross_val_score(model, x_train, y_train).mean())