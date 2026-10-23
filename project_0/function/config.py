from pathlib import Path

#----------路径填写------------
#原始路径目录，即全部数据存放的目录
DATA_DIR = Path(r'D:\系统python学习\机器学习\Corporación Favorita商店销量预测项目\Corporación Favorita商店销量预测项目')

#结果输出路径,路径仅存于内存
ARTIFACT_DIR= Path(__file__).resolve().parents[1] / 'artifacts'#魔术封装方法__file__得到当前文件路径，获取当前文件的绝对路径resolve（全名），parents[1]获取上一级目录
#建立artifacts目录
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
#模型保存路径,仅仅存在与内存中，后续用函数保存为.pkl文件
MODEL_DIR = ARTIFACT_DIR / 'model.pkl'
#结果生成路径，后续用函数保存为.csv文件
SUBMISSION_DIR = ARTIFACT_DIR / 'submission.csv'

#----------任务目的------------
TARGET = 'sales'                                      #预测目标变量名称
TIME_COLUMN = 'date'                                  #时间变量名称，时间序列排序
KEYS = ['store_nbr', 'family']                        #train数据中的变量名，作为主键，将门店和品类结合，同一门店卖的同一种商品，销量随着时间推移
HORIZON = 10                                          #预测 horizon 天的销量

#----------类别变量------------静态数据
CATEGORICAL = ['store_nbr','family','state','type','cluster','city']#告诉模型这些变量是类别变量，不要将里面的值转换为数值型，不应该用门店序号大小做排序，cluster为两个门店顾客消费习惯，节假日波动，促销反应相似，归为一类，补充跨地域的相似性。oil作为未来未知量，二次预测会导致模型过拟合，油价使用滞后填补。

#----------特征参数-------------
LAGS = [10,14,21,28]                                  #滞后天数,均大于等于horizon，同时尽量用7的倍数对齐星期
ROLLING = [7,14,21]                                   #滚动窗口
ROLLING_SHIFT = HORIZON                               #滚动前先shift，滞后horizon天

#----------时间切分做早停-------------
VALID_DAYS = HORIZON                                  #验证集天数，取得是最后horizon的天数

# ---------- 模型参数 ----------
SEED = 2026
# 待调参数
LGB_PARAMS = {
    "objective": "regression_l1",      # MAE，对偏斜销量更稳
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 0,
    "seed": SEED,
    "verbose": -1,
}
NUM_BOOST_ROUND = 3000
EARLY_STOPPING = 100






