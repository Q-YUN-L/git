#特征处理

from . import config

#时间序列特征处理
def time_features(df):
    d = df[config.TIME_COLUMN].dt
    df['year'] = d.year
    df['month'] = d.month
    df['day'] = d.day
    df['weekday'] = d.weekday
    # 0/1转换,月初，月末，周末，发薪水日
    df['is_month_start'] = d.is_month_start.astype(int)
    df['is_month_end'] = d.is_month_end.astype(int)
    df['is_weekend'] = (d.weekday >= 5).astype(int)
    df['is_payday']  =((df['day']==15)|(df['is_month_end']==1)).astype(int)
    return df

#---------------------外部表处理-------------------------
#oil处理
def prepare_oil(oil):
    #油价排序,设置索引,重采样到每日
    s = oil.set_index('date')["dcoilwtico"].resample('D').mean()
    #前后补充缺失值
    s = s.interpolate().ffill().bfill()
    #输出结果重命名
    out = s.rename("oil").reset_index()
    #添加新的滞后列,和30滚动平均列
    out['oil_lag'] = out['oil'].shift(config.HORIZON)
    out['oil_roll30_mean'] = out['oil'].shift(config.HORIZON).rolling(window=30).mean()
    return out.drop(columns=['oil'])

#holiday处理
def prepare_holidays(holidays):
    #过滤调休,筛选出非调休的假期
    h = holidays[~holidays['transferred']]
    #去重复
    flag = h[['date']].drop_duplicates().assign(is_holiday=1)
    return flag

#表格合并
def merge_ex(df,stores,oil_daily,holiday_daily):
    df = df.merge(stores, on='store_nbr', how='left')
    df = df.merge(oil_daily, on='date', how='left')
    df = df.merge(holiday_daily, on='date', how='left')
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    #按时间排序,前向填充补充油价
    df = df.sort_values(by='date')
    df[['oil_lag','oil_roll30_mean']] = df[['oil_lag','oil_roll30_mean']].ffill()
    return df

#历史特征处理
def history_add(df):
    #按照时间和特征先进行排序
    df = df.sort_values([config.TIME_COLUMN]+config.KEYS).reset_index(drop=True)
    #分组取y
    g = df.groupby(config.KEYS,sort=False)[config.TARGET]
    #对齐滞后特征
    for i in config.LAGS:
        df[f'lag_{i}'] = g.shift(i)
    #滚动统计,先滞后再滚动
    s = config.ROLLING_SHIFT
    for i in config.ROLLING:
        df[f'rolling_mean{i}'] =g.transform(lambda x: x.shift(s).rolling(window=i).mean())
        df[f'rolling_std{i}'] = g.transform(lambda x: x.shift(s).rolling(window=i).std())
    return df


#------入口操作,特征处理入口
def build_features(df,oil,holidays,stores):
    """
    df:数据
    oil:油价
    holidays:假期
    stores:商店信息
    """
    df = df.copy()
    #时间处理
    df = time_features(df)
    #表格合并，内容自带oil,holiday处理
    df = merge_ex(df,stores,prepare_oil(oil),prepare_holidays(holidays))
    #历史特征处理
    df = history_add(df)
    #对全部特征进行category处理，供lightbgm使用
    for c in config.CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype('category')
    return df

#选出入模的列
def feature_select(df):
    #丢弃列
    drop = ['id',config.TIME_COLUMN,config.TARGET,'sales_clip']#sales_clip为截断，把大于上限的值强制压到上限，小于下限压到下限
    return [c for c in df.columns if c not in drop]



















