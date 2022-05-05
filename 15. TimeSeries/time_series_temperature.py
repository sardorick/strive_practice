import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn import pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

data = pd.read_csv('15. TimeSeries\climate.csv')
data = data.drop(['Date Time'], axis=1)

def pairing(data, seq_len=6):
    x = []
    y = []

    for i in range(0, (data.shape[0] - seq_len+1), seq_len+1):
        seq = np.zeros((seq_len, data.shape[1]))

        for j in range(seq_len):
            seq[j] = data.values[i+j]

        x.append(seq.flatten())
        y.append(data["T (degC)"][i+seq_len])
    return np.array(x), np.array(y)

print(data.shape)

x, y = pairing(data)
# x_df = pd.DataFrame(x)
# print(x.shape)
# print(y.shape)

# def getfeatures(data):

#     # for holding extracted features
#     new_data = []
#     # get each group
#     for i in range(data.shape[0]):

#         group = []   # to hold extracted elements from each column
#         names = []
#         # get each column within each group
#         for j in range(data.shape[2]):

#             group.append(np.mean(data[i][:, j]))  # mean
#             group.append(np.std(data[i][:, j]))  # standard deviation
#             group.append(data[i][:, j][-1])      # last element

#         new_data.append(group)

#     return np.array(new_data)              



x_train, y_train = x[:45000], y[:45000]
x_test, y_test = x[45000:], y[45000:]

# model = RandomForestRegressor()

# model.fit(x_train, y_train)

# predictions = model.predict(x_test)
# print(predictions)

models_reg = {
  "Extra Trees":   ExtraTreesRegressor(random_state=0),
  "Random Forest": RandomForestRegressor(random_state=0),
  "AdaBoost":      AdaBoostRegressor(random_state=0),
  "Skl GBM":       GradientBoostingRegressor(random_state=0),
}

models = {name: pipeline.make_pipeline(model) for name, model in models_reg.items()}

results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], 'Time': []})

for model_name, model in models_reg.items():
    # iterate through every model in the model dict and train and predict
    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time
        
    pred = model.predict(x_test)
    # Each model result
    results = results.append({"Model":    model_name,
                              "MSE": mean_squared_error(y_test, pred),
                              "MAB": mean_absolute_error(y_test, pred),
                              "Time":     total_time},
                              ignore_index=True)


results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
print(results_ord)

"""
           Model       MSE       MAB        Time
0    Extra Trees  0.046142  0.139590   76.331506
1  Random Forest  0.046882  0.140182  238.846753
2        Skl GBM  0.062394  0.173206   76.581876
3       AdaBoost  0.361889  0.463528   38.058786
"""