import joblib
import optuna
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB


def __apply_smote(X, y):
    """ SMOTE 적용 (훈련 데이터에서만 실행) """
    smote = SMOTE(sampling_strategy=0.65, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def optimize_xgb(trial, X, y):
    """ Optuna를 이용한 XGBoost 하이퍼파라미터 최적화 """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
    }
    model = xgb.XGBClassifier(**params, use_label_encoder=False)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(
            X_train, y_train
            , eval_set=[(X_val, y_val)]  
            #, eval_metric="logloss"
            #, early_stopping_rounds=10  # ✅ 최신 XGBoost에 맞게 수정
            , verbose=False
            
        )
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds))

    return np.mean(scores)


def train_model(X, y):
    """ 모델 학습 및 평가 """
    X_train_resampled, y_train_resampled = __apply_smote(X, y)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optimize_xgb(trial, X_train_resampled, y_train_resampled), n_trials=20)

    best_xgb_params = study.best_params
    xgb_model = xgb.XGBClassifier(**best_xgb_params, use_label_encoder=False)

    gbc = GradientBoostingClassifier(n_estimators=300)
    rf = RandomForestClassifier(n_estimators=300)
    ridge = RidgeClassifier()
    nb = GaussianNB()

    stack_model = StackingClassifier(
        estimators=[("gbc", gbc), ("rf", rf), ("xgb", xgb_model), ("ridge", ridge), ("nb", nb)],
        final_estimator=LogisticRegression()
    )

    stack_model.fit(X_train_resampled, y_train_resampled)

    joblib.dump(stack_model, "models/best_model.pkl")
    print("✅ Best Stacking Model saved as best_model.pkl")

    return stack_model


def run_training(X_train, y_train):
    return train_model(X_train, y_train)
