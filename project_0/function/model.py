from . import config
#保存和加载模型
import joblib

import numpy as np
import lightgbm as lgb

# 创建类对象
class SaleForecaster:
    def __init__(self):
        self.model = None #训练完毕，使用lgb.Booster
        self.features = None #训练用到的特征列

    def fit(self,X_train,y_train,X_valid,y_valid):   #标准化，训练模型
        self.features = list(X_train.columns)
        #防御性获取类别特征，防止训练集和验证集中存在测试集中不存在的类别特征
        cat = [c for c in config.CATEGORICAL if c in self.features]

        dtrain = lgb.Dataset(X_train,np.log1p(y_train),categorical_feature=cat)
        dvalid = lgb.Dataset(X_valid,np.log1p(y_valid),reference=dtrain)

        self.model = lgb.train(
            config.LGB_PARAMS,
            dtrain,
            valid_sets=[dvalid],
            num_boost_round=config.NUM_BOOST_ROUND,
            valid_names=['valid'],
            callbacks=[
                lgb.early_stopping(config.EARLY_STOPPING),
                lgb.log_evaluation(100)
            ]
        )
        #返回self的内容，包含self.model,self.features
        return self

    # 未来预测
    def predict(self,X):
        p = self.model.predict(X[self.features],num_iteration=self.model.best_iteration)
        #best_iteration只用早停找到的最优迭代轮次做预测
        return np.clip(np.expm1(p),0,None)

    #模型保存
    def save(self,path = config.MODEL_DIR):
        joblib.dump({"model":self.model,"features":self.features},path)

    #调用该类方法
    @classmethod
    #模型加载，cls为该类的实例，cls 为 class SaleForecaster的self，创建一个实例obj，同上述self.model,self.features调用
    def load(cls,path = config.MODEL_DIR):
        obj = cls()
        d = joblib.load(path)
        obj.model = d["model"]
        obj.features = d["features"]
        return obj
