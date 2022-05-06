import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from sklearn.model_selection import train_test_split
np.random.seed(0)

data = pd.read_csv('15. TimeSeries\climate.csv')

data = data.drop(['Date Time'], axis=1)
# print(data.corrwith(data['T (degC)']).sort_values(ascending=False))
"""
T (degC)           1.000000
Tpot (K)           0.996827
VPmax (mbar)       0.951113
Tdew (degC)        0.895708
VPact (mbar)       0.867673
H2OC (mmol/mol)    0.867177
sh (g/kg)          0.866755
VPdef (mbar)       0.761744
wd (deg)           0.038732
max. wv (m/s)     -0.002871
wv (m/s)          -0.004689
p (armb)          -0.045375
rh (%)            -0.572416
rho (g/m**3)      -0.963410
"""
data = data.drop(columns=["wd (deg)", "max. wv (m/s)", "wv (m/s)", "p (armb)"])
def get_sequence(data, target_name,  seq_len=6):

    seq_list = []
    target_list = []

    for i in range(0, data.shape[0] - (seq_len+1), seq_len+1):

        seq = data[i: seq_len + i]
        target = data[target_name][seq_len + i]

        seq_list.append(seq)
        target_list.append(target)

    return np.array(seq_list), np.array(target_list)


x, y = get_sequence(data, target_name='T (degC)')
# x_df = pd.DataFrame(x)
# print(x.shape)
# print(y.shape)

def get_features(data):

    features = []
    # iterate through each row of data
    for i in range(data.shape[0]):
        each_column = []
        for j in range(data.shape[2]):
            col_j = np.mean(data[i][:,j])
            each_column.append(col_j)
            # col_2 = np.mean(data[i][:,1])
            # col_3 = np.mean(data[i][:,2])
            # col_4 = np.std(data[i][:,3])
            # col_5 = np.std(data[i][:, 4])
            # col_6 = np.mean(data[i][:,5])
            # col_7 = np.mean(data[i][:,6])
            # col_8 = np.mean(data[i][:,7])
            # col_9 = np.std(data[i][:,8])
            # col_10 = np.std(data[i][:, 9])
            # col_11 = np.mean(data[i][:,10])

        
        features.append([each_column])

    return np.array(features).reshape((60078, 10))              

x = get_features(x)
# print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

scaler = pipeline.Pipeline(steps=[ ('scaler', StandardScaler()) ])
ctr = ColumnTransformer(transformers=[('num', scaler)], remainder="passthrough")
models_reg = {
  "Extra Trees":   ExtraTreesRegressor(random_state=0),
  "Random Forest": RandomForestRegressor(random_state=0),
  "AdaBoost":      AdaBoostRegressor(random_state=0),
  "Skl GBM":       GradientBoostingRegressor(random_state=0),
}

models = {name: pipeline.make_pipeline(ctr, model) for name, model in models_reg.items()}

results = pd.DataFrame({'Model': [], 'R2': [], 'MAB': [], 'Time': []})

for model_name, model in models_reg.items():
    # iterate through every model in the model dict and train and predict
    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time
        
    pred = model.predict(x_test)
    # Each model result
    results = results.append({"Model":    model_name,
                              "R2": r2_score(y_test, pred),
                              "MAB": mean_absolute_error(y_test, pred),
                              "Time":     total_time},
                              ignore_index=True)


results_ord = results.sort_values(by=['R2'], ascending=False, ignore_index=True)
print(results_ord)

# With all features mean and std
"""
           Model       MSE       MAB        Time
0    Extra Trees  0.046142  0.139590   76.331506
1  Random Forest  0.046882  0.140182  238.846753
2        Skl GBM  0.062394  0.173206   76.581876
3       AdaBoost  0.361889  0.463528   38.058786
"""
# With 10 features and mean for each feature
"""
           Model        R2       MAB       Time
0       AdaBoost  0.989156  0.663890   2.955671
1    Extra Trees  0.993828  0.478168  11.211000
2  Random Forest  0.994175  0.460484  33.182516
3        Skl GBM  0.994565  0.442818  10.634789
"""

# With 10 features and std
"""
0    Extra Trees  0.991407  0.572964  10.259279
1  Random Forest  0.975915  0.920843  26.773253
2        Skl GBM  0.962449  1.177591  11.089085
3       AdaBoost  0.839446  2.663351   4.398919
"""
# With scaled columns and mean for all 10 of them
"""
           Model        R2       MAB       Time
0        Skl GBM  0.994565  0.442818  10.576430
1  Random Forest  0.994175  0.460484  32.676212
2    Extra Trees  0.993828  0.478168  11.387216
3       AdaBoost  0.989156  0.663890   2.999767
"""