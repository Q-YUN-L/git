import pandas as pd

#当前项目文件路径方法导入
from . import config


#封装函数方法
def LOADING():
    #目录绝对路径
    d = config.DATA_DIR
    #目录全数据路径
    oil = pd.read_csv(d / "oil.csv" , parse_dates=["date"]) #pd化数据，parse_date将csv中的时间格式变为pd能解析的2000-01-01格式
    stores = pd.read_csv(d / "stores.csv")
    holidays = pd.read_csv(d / "holidays_events.csv",parse_dates=["date"])
    test = pd.read_csv(d / "test.csv",parse_dates=["date"])
    train = pd.read_csv(d / "train.csv",parse_dates=["date"])
    return oil, stores, holidays, test, train
