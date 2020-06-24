'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-05-22 14:45:05
@LastEditors: Troy Wu
@LastEditTime: 2020-06-24 17:14:29
'''
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
# import shap
from catboost import CatBoostRegressor
from sklearn import linear_model
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, RandomForestRegressor)
from sklearn.feature_selection import (SelectFromModel, SelectKBest,
                                       f_regression, mutual_info_regression)
from sklearn.linear_model import (BayesianRidge, Lars, Lasso, LinearRegression,
                                  Ridge)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler,
                                   PolynomialFeatures)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# 读取数据
path = r'D:\troywu666\personal_stuff\二手车交易价格预测\\'
train = pd.read_csv(path + 'used_car_train_20200313.csv', sep=' ')
test = pd.read_csv(path + r'used_car_testB_20200421.csv', sep=' ')

# 数据探索
for col in train.columns:
    if col != 'price':
        for data in [train, test]:
            print(data[col].value_counts())

# 数据处理
train['price'] = np.log1p(train['price'])
df = pd.concat([train, test], ignore_index=True)
df['name_count'] = df.groupby(['name'])['SaleID'].transform('count')
del df['name']
print('name新特征构造完成')

# 异常值处理
## 该样本的seller特征在test不存在，进行删除
df.drop(df[df['seller'] == 1].index, inplace=True)
df['fuelType'].fillna(0, inplace=True)
df['gearbox'].fillna(0, inplace=True)
df['bodyType'].fillna(0, inplace=True)
df['model'].fillna(0, inplace=True)
df['power'] = df['power'].map(lambda x: 600 if x > 600 else x)
df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', None)
print('异常值处理完成')


# 特征构造
## 时间、地区
def date_process(x):
    year = int(str(x)[:4])
    month = int(str(x)[4:6])
    day = int(str(x)[6:8])
    if month < 1:
        month = 1
    date = datetime(year, month, day)
    return date


df['regDate'] = df['regDate'].apply(date_process)
df['creatDate'] = df['creatDate'].apply(date_process)
df['regDate_year'] = df['regDate'].dt.year
df['regDate_month'] = df['regDate'].dt.month
df['regDate_day'] = df['regDate'].dt.day
df['creatDate_year'] = df['creatDate'].dt.year
df['creatDate_month'] = df['creatDate'].dt.month
df['creatDate_day'] = df['creatDate'].dt.day
df['car_age_day'] = (df['creatDate'] - df['regDate']).dt.days
df['car_age_year'] = round(df['car_age_day'] / 365, 1)
print('时间特征构造完成')

## 分类特征
bin = [i * 20 for i in range(31)]
df['power_bin'] = pd.cut(df['power'], bin, labels=False, include_lowest=True)
df[['power', 'power_bin']]

# bin = [i*10 for i in range(24)]
# df['model_bin'] = pd.cut(df['model'], bin, labels = False)
# df[['model', 'model_bin']]

# def mode(x):
#     return st.mode(x, axis = None, nan_policy = 'omit')[0][0]

# for col in ['regionCode', 'model', 'brand', 'kilometer', 'bodyType', 'fuelType', 'gearbox']:
#     for reg_col in ['price', 'car_age_day', 'power', 'v_0', 'v_1', 'v_2', 'v_3',\
#        'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12',\
#        'v_13', 'v_14']:
#         df[col + '_' + reg_col + '_max'] = df.groupby(col)[reg_col].transform(max)
#         df[col + '_' + reg_col + '_amount'] = df.groupby(col)[reg_col].transform(len)
#         df[col + '_' + reg_col + '_min'] = df.groupby(col)[reg_col].transform(min)
#         df[col + '_' + reg_col + '_median'] = df.groupby(col)[reg_col].transform('median')
#         df[col + '_' + reg_col + '_sum'] = df.groupby(col)[reg_col].transform(sum)
#         df[col + '_' + reg_col + '_mean'] = df.groupby(col)[reg_col].transform('mean')
#         #df[col + '_' + reg_col + '_kurt'] = df.groupby(col)[reg_col].transform(st.kurtosis, axis = None, nan_policy = 'omit')
#         df[col + '_' + reg_col + '_skew'] = df.groupby(col)[reg_col].transform(st.skew, axis = None, nan_policy = 'omit')
#         df[col + '_' + reg_col + '_mad'] = df.groupby(col)[reg_col].transform(st.median_absolute_deviation, axis = None, nan_policy = 'omit')
#         df[col + '_' + reg_col + '_mod']  = df.groupby(col)[reg_col].transform(mode)
# print('分类特征构造完成')

## 特征交叉
for i in range(15):
    for j in range(15):
        if i != j:
            df['new' + str(i) + '+' +
               str(j)] = df['v_' + str(i)] + df['v_' + str(j)]

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(df[['v_0', 'v_1', 'v_2', 'v_3',\
       'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12',\
       'v_13', 'v_14']])
df.drop(['v_0', 'v_1', 'v_2', 'v_3',\
       'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12',\
       'v_13', 'v_14'], axis = 1, inplace = True)
df = pd.concat([df, pd.DataFrame(X_poly, columns=poly.get_feature_names())],
               axis=1)
print('特征交叉构造完成')

feature_list = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'power', 'kilometer', \
    'notRepairedDamage', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', \
    'v_8', 'v_9', 'v_10', 'v_11', 'v_12','v_13', 'v_14']
cla_feature = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', \
    'power_bin']
reg_feature = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', \
    'v_8', 'v_9', 'v_10', 'v_11', 'v_12','v_13', 'v_14']

# 特征筛选
## 特征相关性
sns.heatmap(train[reg_feature].corr())

## 相关性过滤
### F检验
f_selector = SelectKBest(f_regression, k=100)
f_selector.fit(train[reg_feature], train['price'])
transformed_data = f_selector.transform(df[reg_feature])

### 互信息法
mutual_selector = SelectKBest(mutual_info_regression, k=100)
mutual_selector.fit(train[reg_feature], train['price'])
transformed_data = f_selector.trnsform(df[reg_feature])

## 嵌入法
embedded_selector = SelectorFromModel(RandomForestRegressor, threshold=10)
embedded_selector.fit(trian[reg_feature], train['price'])
transformed_data = embedded_selector.transform(df[reg_feature])

## 包裹法
wrapper = RFE(RandomForestRegressor, n_features_to_select=50, step=5)
wrapper.fit(train[reg_feature], train['price'])
transformed_data = wrapper.transform(df[reg_feature])

# 模型构建
train_X = df[feature_list][df['price'].isnull()]
train_y = df['price'][df['price'].isnull()]

## lightgbm
### origin
train_X, test_X, train_y, test_y = train_test_split(train_X,
                                                    train_y,
                                                    test_size=0.25)
lgb_train = lgb.Dataset(train_X, train_y)
lgb_eval = lgb.Dataset(test_X, test_y)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'num_trees': 100,
    'metric': 'mae',
    'max_depth': 5,
    'lambda_l2': 2,
    'min_data_in_leaf': 12,
    'learning_rate': 0.1,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_freq': 1,
    'bagging_seed': 11,
    'bagging_fraction': 0.8,
    'num_leaves': 31
}
lgb_model_1 = lgb.train(params,
                        lgb_train,
                        num_boost_round=20,
                        valid_set=lgb_eval,
                        early_stopping_rounds=10)
predictions = lgb_model_1.predict(test_X,
                                  num_iteration=lgb_model_1.best_iteration)
print('MAE is ', mean_absolute_error(np.expm1(predictions), np.expm1(test_y)))

### sklearn
lgb_model_2 = lgb.LGBRegressor(max_depth=6,
                               num_leaves=31,
                               learning_rate=0.1,
                               n_estimators=100,
                               objective='regression_l1')
lgb_model_2.fit(train_X, train_y)
explainer = shap.TreeExplainer(lgb_model_2)
shap_values = explainer.shap_values(train_X)
shap.summary_plot(shap_values, train_X, plot_type='bar')

## catboost
params = {
    'n_estimators': 200,
    'learning_rate': 0.03,
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'random_seed': 1,
    'depth': 6,
    'use_best_model': True,
    'subsample': 0.6,
    'bootstrap_type': 'Bernoulli',
    'reg_lambda': 3,
    'one_hot_max_size': 2,
    'sampling_frequency': 'PerTree'
}
cab_model = CatBoostRegressor(**params)
cab_model.fit(train_X,
              train_y,
              eval_set=[(test_X, test_y)],
              verbose=300,
              early_stopping_rounds=300)
predictions = cab_model.predict(test_X, ntree_end=cab_model.best_iteration_)
print('MAE is ', mean_absolute_error(np.expm1(predictions), np.expm1(test_y)))

## 贝叶斯岭回归
byr_model = BayesianRidge(n_iter=300)
byr_model.fit(train_X, train_y)
predictions = byr_model.predict(test_X)
print('MAE is ', mean_absolute_error(np.expm1(predictions), no.expm1(test_y)))

## ridge回归
ridge = Ridge(alpha=0.8)
ridge.fit(train_X, train_y)
predictions = ridge.predict(test_X)
print('MAE is ', mean_absolute_error(np.expm1(predictions), np.expm1(test_y)))

## lasso回归
lasso = Lasso(alpha=0.9)
lasso.fit(train_X, train_y)
predictions = lasso.predict(test_X)
print('MAE is ', mean_absolute_error(np.expm1(predictions), np.expm1(test_y)))

## 最小角回归
lars = Lars(n_nozero_coefs=100)
lars.fit(train_X, train_y)
predictions = lars.predict(test_X)
print('MAE is ', mean_absolute_error(np.expm1(predictions), np.expm1(test_y)))

## 线性回归
lr = LinearRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(train_X, train_y)
print('MAE is ', mean_absolute_error(np.expm1(predictions), np.expm1(test_y)))

## 决策树回归
dtr = DecisionTreeRegressor(criterion='mae',
                            max_depth=5,
                            min_samples_split=4,
                            max_features='sqrt',
                            min_samples_leaf=2)
dtr.fit(train_X, train_y)
predictions = dtr.predict(test_X)
print('MAE is ', mean_absolute_error(np.expm1(predictions), np.expm1(test_y)))

## 极端树回归
etr = ExtraTreesRegressor(criterion='mae',
                          max_depth=5,
                          min_samples_split=4,
                          min_samples_leaf=2,
                          max_features='auto')
etr.fit(train_X, train_y)
predictions = etr.predict(test_X)
print('MAE is ', mean_absolute_error(np.expm1(predictions), np.expm1(test_y)))

## 随机森林回归
rfr = RandomForestRegressor(n_estimators=100,
                            criterion='mae',
                            max_features='auto',
                            min_impurity_decrease=1,
                            max_depth=5,
                            min_samples_leaf=2,
                            min_samples_split=4)
rfr.fit(train_X, train_y)
predictions = rfr.predict(test_X)
print('MAE is ', mean_absolute_error(np.expm1(predictions), np.expm1(test_y)))

## adaboost回归
adb = AdaBoostRegressor(b_estimators=50, learning_rate=0.8, loss='linear')
adb.fit(train_X, train_y)
predictions = adb.predict(test_X)
print('MAE is ', mean_absolute_error(np.expm1(predictions), np.expm1(test_y)))

## gbdt回归
gbdt = GradientBoostingRegressor(n_estimators=100,
                                 loss='lad',
                                 learning_rate=0.1,
                                 criterion='mae',
                                 min_samples_leaf=2,
                                 min_samples_split=4,
                                 max_depth=5,
                                 max_features=2,
                                 min_impurity_decrease=1,
                                 max_leaf_nodes=2,
                                 subsample=0.8)
print('MAE is ', mean_absolute_error(np.expm1(predictions), np.expm1(test_y)))
