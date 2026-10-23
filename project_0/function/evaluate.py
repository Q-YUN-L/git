#评估函数
import pandas as pd
import numpy as np

#导包
from . import config


#mape函数替换为wmape，业务痛点：销量数据大量存在`y_true=0`。MAPE 遇到真实值为 0 会除以 0，得到 inf；WMAPE 分母是总和，对大量 0 样本更鲁棒。
def wmape(y_true, y_pred):#加权相对百分比误差
    y_true, y_pred = np.asarray(y_true,float), np.asarray(y_pred,float)#array对数据进行拷贝
    demo = np.abs(y_true).sum()
    return np.abs(y_true-y_pred).sum() / demo if demo else np.nan     #加权计算

def rmsle(y_true, y_pred):#均方对数误差平方根
    y_true, y_pred =np.clip(np.asarray(y_true,float), 0, None), np.clip(np.asarray(y_pred,float), 0, None)#**截断限制**：所有小于 0 的值强制变成 0；大于 0 的保持不变。
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))          #log1p(x) = ln(x+1)
'''
1. 对数变换会**压低大数值的影响**，对小销量样本更加敏感；
2. 适合**非负的预测任务（销量、流量、计数）**；
3. 对 0 值友好，不需要避免 y_true=0；
4. 注意：RMSLE 只衡量**相对误差**，不代表绝对误差。
'''

#总结函数，一次性返回所有指标
def summary(y_true, y_pred):
    #绝对误差的平均值
    mae = np.abs(np.asarray(y_true,float)-np.asarray(y_pred,float)).mean()
    return {'wmape(加权相对百分比误差)':round(wmape(y_true, y_pred),4),
            'rmsle(均方对数误差平方根)': round(rmsle(y_true, y_pred), 4),
            'mae(平均绝对误差)':round(mae,4)}

#基线函数，用来判断模型训练效果，小于该基线则良好，这里用上一周同一天预测今天,eval_mask为布尔判断
def sensonal_naive_wmape(df,eval_mask,lag_days=7):
    hist = df[config.KEYS + [config.TARGET,config.TIME_COLUMN]].copy()#复制，df中括号表示dataframe类型数据
    hist[config.TIME_COLUMN] = hist[config.TIME_COLUMN] + pd.Timedelta(days=lag_days)#7天延时
    hist = hist.rename(columns={config.TARGET: 'naive'})#重命名y
    part = df.loc[eval_mask,config.KEYS + [config.TARGET,config.TIME_COLUMN]].merge(hist,on=config.KEYS+[config.TIME_COLUMN],how='left')#合并,左表df对齐，merge中的hist为右表,how为左对齐
    return wmape(part[config.TARGET], part['naive'])


