import numpy    as np
from numpy.testing._private.utils import decorate_methods
import pandas   as pd
import seaborn  as sb
import matplotlib.pyplot as plt
import sklearn  as skl
import time
from scipy.stats import pearsonr

from sklearn import pipeline      # Pipeline
from sklearn import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection # train_test_split
from sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import set_config

from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import RandomForestRegressor
from sklearn.ensemble      import ExtraTreesRegressor
from sklearn.ensemble      import AdaBoostRegressor
from sklearn.ensemble      import GradientBoostingRegressor
from xgboost               import XGBRegressor
from lightgbm              import LGBMRegressor
from catboost             import CatBoostRegressor

data = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Documents\Strive repos\strive_practice\11. Data Enhancement\data\london_merged.csv")
# print(data.corr()['cnt'].abs().sort_values())
# is_holiday      0.051698
# is_weekend      0.096499
# season          0.116180
# wind_speed      0.116295
# weather_code    0.166633
# t2              0.369035
# t1              0.388798
# hum             0.462901
# cnt             1.000000

np.random.seed(0)

# target = data['cnt']
# data = data.drop(['cnt'], axis=1)

data['year'] = data['timestamp'].apply(lambda row: row[:4])
data['month'] = data['timestamp'].apply(lambda row: row.split('-')[2][:2] )
data['hour'] = data['timestamp'].apply(lambda row: row.split(':')[0][-2:] )
data.drop('timestamp', axis=1, inplace=True)
data['is_dayoff'] = data['is_holiday'] + data['is_weekend']
data.drop(['is_holiday', 'is_weekend'], axis=1, inplace=True)
# Correlation check for various data
def correlation(data, data1):
    corr = pearsonr(data, data1)
    return corr

# print(correlation(data['hum'], data['wind_speed'])) # result is -0.28
# print(correlation(data['is_holiday'], data['is_weekend'])) # result is -0.09
# print(correlation(data['season'], data['wind_speed'])) # result is 0.01

# Enhance data

def data_enhance(data):
    gen_data = data
    for season in data['season'].unique():
        seasonal_data =  gen_data[gen_data['season'] == season]
        hum_median = seasonal_data['hum'].median()
        wind_speed_median = seasonal_data['wind_speed'].median()
        t1_median = seasonal_data['t1'].median()
        t2_median = seasonal_data['t2'].median()
        dayoff_mode = seasonal_data['is_dayoff'].mode() # should we use mode for cat data?
        for i in gen_data[gen_data['season'] == season].index:
            if np.random.randint(2) == 1:
                gen_data['hum'].values[i] += hum_median/4
            else:
                gen_data['hum'].values[i] -= hum_median/4
                
            if np.random.randint(2) == 1:
                gen_data['wind_speed'].values[i] += wind_speed_median/4
            else:
                gen_data['wind_speed'].values[i] -= wind_speed_median/4
                
            if np.random.randint(2) == 1:
                gen_data['t1'].values[i] += t1_median/4
            else:
                gen_data['t1'].values[i] -= t1_median/4
                
            if np.random.randint(2) == 1:
                gen_data['t2'].values[i] += t2_median/4
            else:
                gen_data['t2'].values[i] -= t2_median/4
            if np.random.randint(2) == 1:
                gen_data['is_dayoff'].values[i] += dayoff_mode
            else:
                gen_data['is_dayoff'].values[i] -= dayoff_mode
    return gen_data

# print(data.head(3))
gen = data_enhance(data)
# print(gen.head(3))

y = data['cnt']
x = data.drop(['cnt'], axis=1)

# Dividing into categorical features and numerical features
cat_vars = ['season', 'is_dayoff','year','month','weather_code']
num_vars = ['t1','t2','hum','wind_speed']

# Splitting the data to train and validation sets
x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y,
                                    test_size=0.2,
                                    random_state=0  # Recommended for reproducibility - kept it as it is
                                )

# Applying enhancement to 25% of the data
extra_sample = gen.sample(gen.shape[0] // 4)
x_train = pd.concat([x_train, extra_sample.drop(['cnt'], axis=1 ) ])
y_train = pd.concat([y_train, extra_sample['cnt'] ])

# Transorming the data to make it more like Gaussian-distributed?
transformer = preprocessing.PowerTransformer()
y_train = transformer.fit_transform(y_train.values.reshape(-1,1))
y_val = transformer.transform(y_val.values.reshape(-1,1))



rang = abs(y_train.max()) + abs(y_train.min())
# Creating pipelines for num and cat data, and then creating a transformer pipeline
num_4_treeModels = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='constant', fill_value=-9999)),
])

cat_4_treeModels = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='constant', fill_value='missing')), # constant strategry
    ('ordinal', preprocessing.OrdinalEncoder())  

])

tree_prepro = compose.ColumnTransformer(transformers=[
    ('num', num_4_treeModels, num_vars),
    ('cat', cat_4_treeModels, cat_vars),
], remainder='drop') # Drop other vars not specified in num_vars or cat_vars

# making a dict with differet regression models
tree_classifiers = {
  "Decision Tree": DecisionTreeRegressor(),
  "Extra Trees":   ExtraTreesRegressor(n_estimators=100),
  "Random Forest": RandomForestRegressor(n_estimators=100),
  "AdaBoost":      AdaBoostRegressor(n_estimators=100),
  "Skl GBM":       GradientBoostingRegressor(n_estimators=100),
  "XGBoost":       XGBRegressor(n_estimators=100),
  "LightGBM":      LGBMRegressor(n_estimators=100),
  "CatBoost":      CatBoostRegressor(n_estimators=100),
}

# applying pipe to models by choosing dict values
tree_classifiers = {name: pipeline.make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}

# results df to print results at the end
results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], " % error": [], 'Time': []})

for model_name, model in tree_classifiers.items():
    # iterate through every model in the classifier dict and train and predict
    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time
        
    pred = model.predict(x_val)
    # change results accordingly
    results = results.append({"Model":    model_name,
                              "MSE": metrics.mean_squared_error(y_val, pred),
                              "MAB": metrics.mean_absolute_error(y_val, pred),
                              " % error": metrics.mean_squared_error(y_val, pred) / rang,
                              "Time":     total_time},
                              ignore_index=True)


results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
results_ord.index += 1 
results_ord.style.bar(subset=['MSE', 'MAE'], vmin=0, vmax=100, color='#5fba7d')

# print(results_ord)
