'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-05-18 10:29:13
@LastEditors: Troy Wu
@LastEditTime: 2020-05-22 14:44:40
'''
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import missingno as mso
import seaborn as sns
import lightgbm as lgb
from sklearn.feature_selection import mutual_info_regression as mir, SelectKBest
import scipy.stats as st

path = r'D:\troywu666\personal_stuff\二手车交易价格预测\\'
train = pd.read_csv(path + 'used_car_train_20200313.csv', sep = ' ')
test = pd.read_csv(path + r'used_car_testB_20200421.csv', sep = ' ')
print(train.info())
print(test.info())
set(train.columns) - set(test.columns) #price字段是label

# 分类型变量
cla_cols = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'seller', 'notRepairedDamage', 'offerType']
for data in [train, test]:
    print(data.shape)
    for col in test.columns:
        print(data[col].value_counts(dropna = False))
    for col in test.columns:
        print(data[col].value_counts(dropna = False).sort_index())

# 数据缺失情况
train['notRepairedDamage'] = train['notRepairedDamage'].replace('-', np.nan)
missing_cols = list(train.columns[train.isnull().sum() > 0])
mso.heatmap(train[missing_cols])
mso.matrix(train)
mso.matrix(test)

# 时间类型变量
train['creatDate'].sort_values()
time_cols = ['regDate', 'creatDate']
train[time_cols + ['price']][train['regDate'].apply(lambda x: str(x)[4: 6] == '00')].corr()
train['creatDate'].apply(lambda x: str(x)[4: 6]).value_counts().sort_index()
train['regYear'] = train['regDate'].apply(lambda x: int(str(x)[: 4]))
train['createYear'] = train['creatDate'].apply(lambda x: int(str(x)[: 4]))
train['usedYear'] = train['createYear'] - train['regYear']
for col in time_cols:
    print(col)
    train[col] = pd.to_datetime(train[col], format = "%Y%m%d", errors = 'coerce')
train['userdDays'] = (train['creatDate'] - train['regDate']).dt.days

# 查看数据的个体
train.groupby('name')['name'].count()

# 可视化
## 分类变量
cla_cols = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
for col in cla_cols:
    train[col] = train[col].astype('category')
    if train[col].isnull().any():
        train[col] = train[col].cat.add_categories(['missing'])
        train[col] = train[col].fillna('missing')

def boxplot(x, y, **kwargs):
    sns.boxplot(x = x, y = y)
    x = plt.xticks(rotation = 90)
f = pd.melt(train, id_vars = ['price'], value_vars = cla_cols)
g = sns.FacetGrid(f, col = 'variable', col_wrap = 2, sharex = False, sharey = False, size = 5)
g = g.map(boxplot, 'value', 'price')

def barplot(x, y, **kwargs):
    sns.barplot(x = x, y = y)
    x = plt.xticks(rotation = 90)
f = pd.melt(train, id_vars = ['price'], value_vars = cla_cols)
g = sns.FacetGrid(f, col = 'variable', col_wrap = 2, size = 5, sharex = False, sharey = False)
g = g.map(barplot, 'value', 'price', estimators = np.mean)

def countplot(x, **kwargs):
    sns.countplot(x = x)
    plt.xticks(rotation = 90)
f = pd.melt(train, value_vars = cla_cols)
g = sns.FacetGrid(f, col = 'variable', col_wrap = 2, sharex = False, sharey = False, size = 5)
g = g.map(countplot, 'value')

## 连续型变量
reg_cols = ['power', 'kilometer', 'regYear', 'createYear', 'v_0', 'v_1', 'v_2', 'v_3',
       'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12',
       'v_13', 'v_14']
for col in reg_cols:
    print(col, train[[col, 'price']].corr().values[0][1])
f, ax =  plt.subplots(len(reg_cols), 1, figsize = (8, len(reg_cols) * 8))
for n, col in enumerate(reg_cols):
    sns.distplot(train[col], ax = ax[n], kde_kws = {'bw': 1.5})

h = pd.melt(train, id_vars = ['price'], value_vars = reg_cols)
g = sns.FacetGrid(h, col = 'variable', col_wrap = 2, sharex = False, sharey = False, size = 5)
g = g.map(sns.regplot, 'value', 'price')

h = pd.melt(train, value_vars = reg_cols)
for func in [st.johnsonsu, st.norm, st.lognorm]:
    g = sns.FacetGrid(h, col = 'variable', col_wrap = 4, sharex = False, sharey = False, size = 5)
    g.map(sns.distplot, 'value', kde_kws = {'bw': 1.5}, fit = func, kde = False)

cmap = sns.diverging_palette(220, 10, as_cmap = True)
plt.figure(figsize = (12, 8))
sns.heatmap(train[reg_cols + ['price']].corr().round(2), annot = True, cmap = cmap)
sns.pairplot(train[reg_cols])