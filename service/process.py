import os
import pickle
import optuna
import warnings
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from utils import reset_seeds

warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)

def __run_hpo(X_train, X_test, y_train, y_test):
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_loguniform('gamma', 0.001, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'verbosity': 0
        }

        model = XGBClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test,  y_pred)

        return accuracy
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    study.optimize(objective, n_trials=50)

    return study.best_params

def __train_and_save_model(X_train, X_test, y_train, y_test, best_params, model_save_path):
    best_model = XGBClassifier(
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        n_estimators=best_params['n_estimators'],
        min_child_weight=best_params['min_child_weight'],
        gamma=best_params['gamma'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        verbosity=0
    )

    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    model_filename = os.path.join(model_save_path, 'best_model.pkl')

    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)
    
    return accuracy

@reset_seeds
def train_model(X_train, X_test, y_train, y_test):
    model_save_path = '../model'

    best_params = __run_hpo(X_train, X_test, y_train, y_test)

    accuracy = __train_and_save_model(X_train, X_test, y_train, y_test, best_params, model_save_path)

    return accuracy