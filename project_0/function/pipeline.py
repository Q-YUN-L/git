#项目全导入
import numpy as np
import pandas as pd

from . import config,loader,evaluate,features
from .model import SaleForecaster
#内部私有函数_函数名
def _history_cols():
    cols = [f"lag_{l}"for l in config.LAGS]
    cols += [f"rolling_mean{w}"for w in config.ROLLING]
    cols += [f"rolling_std{w}"for w in config.ROLLING]
    return cols

def run_train():
    print(">>加载数据")
    oil, stores, holidays, test, train = loader.LOADING()

    print(">>构造特征")
    df = features.build_features(train, oil, holidays, stores)
    #删除na行
    df = df.dropna(subset=_history_cols()).reset_index(drop=True)#旧的索引丢弃

    print(">>按照时间切分训练/验证")
    spilt = df[config.TIME_COLUMN].max() - pd.Timedelta(days=config.VALID_DAYS)
    #时间划分
    tr , val = df[df[config.TIME_COLUMN]<=spilt] , df[df[config.TIME_COLUMN]>spilt]
    #入围特征
    feat = features.feature_select(df)
    print(f"train={tr.shape}, valid={val.shape}, features特征数={len(feat)}")

    print('>>训练lightgbm')
    model = SaleForecaster().fit(tr[feat],tr[config.TARGET],val[feat],val[config.TARGET])

    print('>>模型评估(验证集)')
    val_pred= model.predict(val)
    #早停集验证
    base = evaluate.sensonal_naive_wmape(df,df[config.TIME_COLUMN]>spilt)
    print('朴素基准WMAPE',round(base,4))
    print("模型指标：",evaluate.summary(val[config.TARGET],val_pred))

    model.save()
    print(f">>模型保存,{config.MODEL_DIR}")

def run_predict():
    print(">>加载数据")
    oil, stores, holidays, test, train = loader.LOADING()

    print(">>拼接特征【历史+未来预测】")
    #时间，keys，以确定未来的安排
    cols = [config.TIME_COLUMN] + config.KEYS + ["onpromotion"]
    #assign,标记用途
    hist = train[cols].assign(id = np.nan, sales = train[config.TARGET])
    #未来测试数据移除sales（答案）
    fut = test[["id"]+cols].assign(sales=np.nan)
    #过去的数据和未来要预测的值拼接成一张表里，上下拼接（行增加），移除索引
    panel = pd.concat([hist,fut],ignore_index=True)

    print(">>构造特征(口径一致)")
    df = features.build_features(panel, oil, holidays, stores)
    #独立一份，不占用原内存,剔除没有id的信息
    future = df[df["id"].notna()].copy()

    print(">>载入模型")
    model = SaleForecaster().load()
    future["sales"] = model.predict(future)

    out = future[["id","sales"]].sort_values("id")
    out["id"] = out["id"].astype(int)
    out.to_csv(config.SUBMISSION_DIR,index=False)
    print(f">>提交文件保存,{config.SUBMISSION_DIR}{len(out)}行")


if __name__ == "__main__":
    run_train()
    run_predict()


