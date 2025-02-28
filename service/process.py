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
from utils import reset_seeds
param_grid_lr = {
    'C': [1],
    'solver': ['liblinear'],
    'max_iter': [400, 500,600],
    'tol':[1e-3, 1e-4]
}

# Random Forest
param_grid_rf = {
    'n_estimators': [200, 300, 400],
    'max_depth': [20, 40, 50],
    'min_samples_split': [10, 15, 20],
    'min_samples_leaf': [4, 6, 8]
}

# XGBoost
param_grid_xgb = {
    'n_estimators': [100, 300],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [2, 3],
    'subsample': [0.7, 0.8, 0.9]
}
@reset_seeds
def process(x,y):
    is_Regression = False
    """ grid search parameters
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

    print("Best Parameterslr:", grid_search_lr.best_params_)
    print("Best Parametersrf:", grid_search_rf.best_params_)
    print("Best Parametersxgb:", grid_search_xgb.best_params_)
    """
    best_lr = LogisticRegression(C=1,max_iter=400,solver='liblinear',tol=0.001)
    best_rf = RandomForestClassifier(max_depth=20,min_samples_split=10,min_samples_leaf=4,n_estimators=200)
    best_xgb = XGBClassifier(learning_rate=0.1,max_depth=2,n_estimators=300,subsample=0.7)
    
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
        sm = SMOTEENN(sampling_strategy=0.65)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        # 모델 학습
        model.fit(X_train, y_train)
        epoch += 1
        print(f"epoch : {epoch}")
        val_score = f1_score(y_test, model.predict(X_test))
        print(y_train.mean())
        print(f"score: {val_score}")
        if val_score > best_score:
            best_score = val_score
            best_model = model
        else:
            break

    print(best_model.score(x, y))

    return best_model
