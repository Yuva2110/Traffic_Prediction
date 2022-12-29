from sklearn.linear_model import LinearRegression
from mlxtend.regressor import StackingRegressor
from skopt.space import Real, Categorical, Integer
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import seaborn as sns  # 基于matplolib的画图模块
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_csv('data/Train.csv')
data = data.sort_values(
	by=['date_time'], ascending=True).reset_index(drop=True)
last_n_hours = [1, 2, 3, 4, 5, 6]
for n in last_n_hours:
	data[f'last_{n}_hour_traffic'] = data['traffic_volume'].shift(n)
data = data.dropna().reset_index(drop=True)
data.loc[data['is_holiday'] != 'None', 'is_holiday'] = 1
data.loc[data['is_holiday'] == 'None', 'is_holiday'] = 0
data['is_holiday'] = data['is_holiday'].astype(int)

data['date_time'] = pd.to_datetime(data['date_time'])
data['hour'] = data['date_time'].map(lambda x: int(x.strftime("%H")))
data['month_day'] = data['date_time'].map(lambda x: int(x.strftime("%d")))
data['weekday'] = data['date_time'].map(lambda x: x.weekday()+1)
data['month'] = data['date_time'].map(lambda x: int(x.strftime("%m")))
data['year'] = data['date_time'].map(lambda x: int(x.strftime("%Y")))
data.to_csv("traffic_volume_data.csv", index=None)
data.columns
# 导入常用python数据分析库
sns.set()  # seaborn美化样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签 
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
data = pd.read_csv("traffic_volume_data.csv")
# data = data[data['year']==2016].copy().reset_index(drop=True)
data = data.sample(10000).reset_index(drop=True)
label_columns = ['weather_type', 'weather_description']
numeric_columns = ['is_holiday', 'air_pollution_index', 'humidity',
                   'wind_speed', 'wind_direction', 'visibility_in_miles', 'dew_point',
                   'temperature', 'rain_p_h', 'snow_p_h', 'clouds_all', 'weekday', 'hour', 'month_day', 'year', 'month', 'last_1_hour_traffic',
                   'last_2_hour_traffic', 'last_3_hour_traffic']
from sklearn.preprocessing import OneHotEncoder
ohe_encoder = OneHotEncoder()
x_ohehot = ohe_encoder.fit_transform(data[label_columns])
ohe_features = ohe_encoder.get_feature_names()
x_ohehot = pd.DataFrame(x_ohehot.toarray(),
						columns=ohe_features)
data = pd.concat([data[['date_time']],data[['traffic_volume']+numeric_columns],x_ohehot],axis=1)
data['traffic_volume'].hist(bins=20)
metrics = ['month', 'month_day', 'weekday', 'hour']

fig = plt.figure(figsize=(8, 4*len(metrics)))
for i, metric in enumerate(metrics):
	ax = fig.add_subplot(len(metrics), 1, i+1)
	ax.plot(data.groupby(metric)['traffic_volume'].mean(), '-o')
	ax.set_xlabel(metric)
	ax.set_ylabel("Mean Traffic")
	ax.set_title(f"Traffic Trend by {metric}")
plt.tight_layout()
plt.show()
# SVR
# 人工神经网络
# stacking
# xgboost
#from xgboost import XGBRegressor
features = numeric_columns+list(ohe_features)
target = ['traffic_volume']
X = data[features]
y = data[target]
x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(y).flatten()
# X = zscore(X)
warnings.filterwarnings('ignore')
##################
regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)
print(regr.predict(X[:10]))
print(y[:10])
##################
# Model_params = {
#     MLPRegressor: {
#         'hidden_layer_sizes': Integer(50, 400),
#         'solver': Categorical(['sgd', 'adam']),
#         'learning_rate_init': Real(0.001, 0.1)
#     },
#     # XGBRegressor: {
#     #     'n_estimators': Integer(10, 30),
#     #     'max_depth': Integer(4, 8),
#     #     'max_leaves': Integer(2, 8)
#     # },
#     SVR: {
#         'C': Real(1e-1, 1e+1, prior='log-uniform'),  # 如果要loguniform
#         'degree': Integer(1, 4),
#         'kernel': Categorical(['linear', 'rbf'])  # Categorical类型的空间
#     }
# }


# def bayes_opt(Model, params: dict, X, y, n_iter=5, scoring='r2', cv=3):
#     """
#     BayesSearch One Model
#     params
#     - Model - Model Class
#     """
#     print(f"- Bayes Optimizing: {Model.__name__}")
#     opt = BayesSearchCV(
#         Model(),
#         params,
#         n_iter=n_iter,
#         scoring=scoring,
#         cv=cv,
#         verbose=1

#     )
#     opt.fit(X, y)
#     print(opt.best_params_)
#     # print(opt.best_score_)
#     return opt, opt.best_params_, opt.best_score_


# best_params_dic = {}
# for Model, params in Model_params.items():
#     opt, best_params_,best_score_ = bayes_opt(Model,params,X,y)
#     best_params_dic[Model] = dict(best_params_)
# best_params_dic