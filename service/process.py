from imblearn.combine import SMOTEENN
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def get_cross_validation(shuffle:bool=True, is_kfold:bool=True, n_splits:int=5):
    if is_kfold:
        return KFold(n_splits=n_splits, shuffle=shuffle)
    else:
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle)

def run_cross_validation(my_model, x_train, y_train, cv, is_kfold:bool=True):
    n_iter = 0
    accuracy_lst = []
    if is_kfold:
        cross_validation = cv.split(x_train)
    else:
        cross_validation = cv.split(x_train, y_train)

    for train_index, valid_index in cross_validation:
        n_iter += 1
        # 학습용, 검증용 데이터 구성
        train_x, valid_x = x_train.iloc[train_index], x_train.iloc[valid_index]
        train_y, valid_y = y_train.iloc[train_index], y_train.iloc[valid_index]
        # 학습
        my_model.fit(train_x, train_y)
        # 예측
        pred = my_model.predict(valid_x)
        # 평가
        accuracy = np.round(accuracy_score(valid_y, pred), 4)
        accuracy_lst.append(accuracy)
        print(f'{n_iter} 번째 K-fold 정확도: {accuracy}, 학습데이터 크기: {train_x.shape}, 검증데이터 크기: {valid_x.shape}')

    return np.mean(accuracy_lst)


def process(x,y):
    is_Regression = False
    model = XGBClassifier(learning_rate= 0.01,max_depth = 3,n_estimators = 1000)
    my_cv = get_cross_validation(is_kfold=is_Regression)
    run_cross_validation(model, x, y, my_cv, is_kfold=is_Regression)
    print(model.score(x, y))

    return model