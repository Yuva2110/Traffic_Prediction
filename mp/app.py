from sklearn.preprocessing import OneHotEncoder
from flask import Flask, render_template, request
from sklearn import svm
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
from datetime import datetime, timedelta


def posix_time(dt):
    return (dt - datetime(1970, 1, 1)) / timedelta(seconds=1)


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
# data.columns
sns.set()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
data = pd.read_csv("traffic_volume_data.csv")
data = data.sample(10000).reset_index(drop=True)
label_columns = ['weather_type', 'weather_description']
numeric_columns = ['is_holiday',  'temperature',
                   'weekday', 'hour', 'month_day', 'year', 'month']
# ohe_encoder = OneHotEncoder()
# x_ohehot = ohe_encoder.fit_transform(data[label_columns])
# ohe_features = ohe_encoder.get_feature_names()
# x_ohehot = pd.DataFrame(x_ohehot.toarray(),
#                         columns=ohe_features)
# data = pd.concat(
#     [data[['date_time']], data[['traffic_volume']+numeric_columns], x_ohehot], axis=1)
# data['traffic_volume'].hist(bins=20)
# metrics = ['month', 'month_day', 'weekday', 'hour']

# fig = plt.figure(figsize=(8, 4*len(metrics)))
# for i, metric in enumerate(metrics):
# 	ax = fig.add_subplot(len(metrics), 1, i+1)
# 	ax.plot(data.groupby(metric)['traffic_volume'].mean(), '-o')
# 	ax.set_xlabel(metric)
# 	ax.set_ylabel("Mean Traffic")
# 	ax.set_title(f"Traffic Trend by {metric}")
# plt.tight_layout()
# plt.show()

#features = numeric_columns+list(ohe_features)
features = numeric_columns
target = ['traffic_volume']
X = data[features]
y = data[target]
x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(y).flatten()
warnings.filterwarnings('ignore')
##################
regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)
new = []
 print(regr.predict(X[:10]))
 print(y[:10])
########################################################################################################
app = Flask(__name__, static_url_path='')


@app.route('/')
def root():
    return render_template('index.html')


d = {}


@app.route('/predict', methods=['POST'])
def predict():
    d['is_holiday'] = request.form['isholiday']
    if d['is_holiday'] == 'yes':
        d['is_holiday'] = int(1)
    else:
        d['is_holiday'] = int(0)
    d['temperature'] = int(request.form['temperature'])
    d['weekday'] = int(0)
    D = request.form['date']
    d['hour'] = int(request.form['time'][:2])
    d['month_day'] = int(D[8:])
    d['year'] = int(D[:4])
   # should change
    d['month'] = int(D[5:7])
    d['x0'] = request.form.get('x0')
    #d['y'] = request.form.get('y')
    d['x1'] = request.form.get('x1')
    # #DATE = request.form['time']
    xoval = {'x0_Clear', 'x0_Clouds', 'x0_Drizzle', 'x0_Fog', 'x0_Haze',
             'x0_Mist', 'x0_Rain', 'x0_Smoke', 'x0_Snow', 'x0_Thunderstorm'}
    x1val = {'x1_Sky is Clear',
             'x1_broken clouds',
             'x1_drizzle',
             'x1_few clouds',
             'x1_fog',
             'x1_haze',
             'x1_heavy intensity drizzle',
             'x1_heavy intensity rain',
             'x1_heavy snow',
             'x1_light intensity drizzle',
             'x1_light intensity shower rain',
             'x1_light rain',
             'x1_light rain and snow',
             'x1_light shower snow',
             'x1_light snow',
             'x1_mist',
             'x1_moderate rain',
             'x1_overcast clouds',
             'x1_proximity shower rain',
             'x1_proximity thunderstorm',
             'x1_proximity thunderstorm with drizzle',
             'x1_proximity thunderstorm with rain',
             'x1_scattered clouds',
             'x1_shower drizzle',
             'x1_sky is clear',
             'x1_sleet',
             'x1_smoke',
             'x1_snow',
             'x1_thunderstorm',
             'x1_thunderstorm with heavy rain',
             'x1_thunderstorm with light drizzle',
             'x1_thunderstorm with light rain',
             'x1_thunderstorm with rain',
             'x1_very heavy rain'
             }
    # print(xoval)
    x0 = {}
    x1 = {}
    for i in xoval:
        x0[i] = 0
    for i in x1val:
        x1[i] = 0
    x0[d['x0']] = 1
    x1[d['x1']] = 1
    # print(x0)
    # print(x1)
    # print(x0)
    # print(d)
    final = []
    final.append(d['is_holiday'])
    final.append(d['temperature'])
    final.append(d['weekday'])
    final.append(d['hour'])
    final.append(d['month_day'])
    final.append(d['year'])
    final.append(d['month'])
    for i in x0:
        final.append(x0[i])
    for i in x1:
        final.append(x1[i])
    # print(d)
    # print(len(final))
    output = print(regr.predict([final]))
    print(output)
    return render_template('output.html', data1=d, data2=final)
if __name__ == '__main__':
    app.run(debug=True)
