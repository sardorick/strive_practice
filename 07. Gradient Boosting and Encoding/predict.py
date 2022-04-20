
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import data_handler as dh
x_train, x_test, y_train, y_test = dh.get_data(r"C:\Users\Lenovo\OneDrive\Documents\Strive repos\strive_practice\07. Gradient Boosting and Encoding\insurance.csv")


model = GradientBoostingRegressor(learning_rate=0.01, max_depth=3, n_estimators=500)

scaler = StandardScaler()
inputs = None
def get_input():
    global inputs
    inputs = []
    age = int(input("How old are you? \n"))
    child = int(input("How many children do you have? \n"))
    smoke = bool(input("Do you smoke? \n"))
    sex = int(input("What is your sex? \n"))
    bmi = float(input("What is your BMI? \n"))
    region = int(input("Choose one of the regions: \n southwest: 1\n southeast: 2\n northwest: 3\n northeast: 4\n"))
    inputs.append([0, age, sex, bmi, child, smoke, region])

    return np.array(inputs)

def predict_input(inps):
    # ct = ColumnTransformer( [('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1), [4,5,6] )], remainder='passthrough')
    # x_train = ct.fit_transform(x_train)
    # y_train = ct.transform(y_train)

    model.fit(x_train, y_train)
    predictions = model.predict(inps)
    # predict
    return predictions

print(predict_input(get_input()))
# print(get_input())
