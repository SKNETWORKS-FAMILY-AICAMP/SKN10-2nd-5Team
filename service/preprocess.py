import os
import numpy as np
import random
import torch
import scipy.stats as stats
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import koreanize_matplotlib  # matplotlib 한글화
import easydict

from service.utils import reset_seeds
#from utils import reset_seeds

warnings.filterwarnings(action='ignore')

# 시드 고정 적용 (수정된 부분)
reset_seeds(seed=42)

def read_csv():
    args = easydict.EasyDict()
    # path info
    args.default_path = "data/"
    args.all_csv = args.default_path + "Telco_customer.csv"
    ori_all = pd.read_csv(args.all_csv)
    return ori_all

###################################### Data Cleaning #########################################

# yes, no인 범주형 컬럼을 0과 1로 변환하는 함수
def binary_categorical_to_numeric(df):
    binary_categorical_columns = ['Partner', 'Dependents', 'PaperlessBilling']
    for column in binary_categorical_columns:
        df[column] = df[column].apply(lambda x: 1 if x == "Yes" else 0)
    return df

# 컬럼 삭제 함수
def drop_columns(df):
    drop_columns = ["customerID", "MonthlyCharges", "PhoneService", "StreamingTV", "gender"]
    df = df.drop(columns=drop_columns, axis=1)
    return df

# lightGBM은 str를 인식 못하므로 category로 변환하는 함수
def str_to_category(df):
    columns = ['InternetService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingMovies', 'Contract', 'PaymentMethod']
    for col in columns:
        df[col] = df[col].str.lower()
        df[col] = df[col].astype("category")
    return df
