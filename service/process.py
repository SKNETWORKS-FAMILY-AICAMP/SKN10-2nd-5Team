from imblearn.combine import SMOTEENN
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier
import matplotlib.pyplot as plt 
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga'],
    'max_iter': [400, 500],
    'tol':[1e-3, 1e-2, 1e-1]
}

# Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}
def process(x,y):
    is_Regression = False

    clf1 = LogisticRegression(random_state=42)
    clf2 = RandomForestClassifier(random_state=42)
    clf3 = XGBClassifier(random_state=42)

    grid_search_lr = GridSearchCV(clf1, param_grid_lr, cv=3, scoring='f1')
    grid_search_rf = GridSearchCV(clf2, param_grid_rf, cv=3, scoring='f1')
    grid_search_xgb = GridSearchCV(clf3, param_grid_xgb, cv=3, scoring='f1')

    grid_search_lr.fit(x, y)
    grid_search_rf.fit(x, y)
    grid_search_xgb.fit(x, y)

    best_lr = grid_search_lr.best_estimator_
    best_rf = grid_search_rf.best_estimator_
    best_xgb = grid_search_xgb.best_estimator_

    # Hard VotingClassifier 정의
    model = VotingClassifier(estimators=[('lr', best_lr), ('rf', best_rf), ('xgb', best_xgb)], voting='hard')
    cv = StratifiedKFold(n_splits=3, shuffle=True)
    # StratifiedKFold Cross Validation 수행
    best_model = None
    best_model = model
    best_score = -float('inf')
    epoch = 0
    

    for train_index, test_index in cv.split(x, y):

        # score 안좋아지면 stop
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        sm = SMOTEENN(sampling_strategy=0.5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        # 모델 학습
        model.fit(X_train, y_train)
        epoch += 1
        print(f"epoch : {epoch}")
        val_score = f1_score(y_test, model.predict(X_test))
        print(f"score: {val_score}")
        if val_score > best_score:
            best_score = val_score
            best_model = model
        else:
            break

    print(best_model.score(x, y))

    return best_model
