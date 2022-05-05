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
# def extract_feature(df):
#     features = []
#     features.append(np.std(df.values[:, 0]))
#     features.append(np.std(df.values[:, 0]))                 
#     features.append(np.std(df.values[:, 0]))
#     features.append(np.std(df.values[:, 0]))                  
#     features.append(np.std(df.values[:, 0]))
#     features.append(np.std(df.values[:, 0]))                  
#     features.append(np.std(df.values[:, 0]))
#     features.append(np.std(df.values[:, 0]))                  
#     features.append(np.std(df.values[:, 0]))
#     features.append(np.std(df.values[:, 0]))                  



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
    # iterate through every model in the classifier dict and train and predict
    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time
        
    pred = model.predict(x_test)
    # change results accordingly
    results = results.append({"Model":    model_name,
                              "MSE": mean_squared_error(y_test, pred),
                              "MAB": mean_absolute_error(y_test, pred),
                              "Time":     total_time},
                              ignore_index=True)


results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
print(results_ord)
