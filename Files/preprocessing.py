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
warnings.filterwarnings(action='ignore')

def reset_seeds(seed=42):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)    # 파이썬 환경변수 시드 고정
  np.random.seed(seed)
  torch.manual_seed(seed) # cpu 연산 무작위 고정
  torch.cuda.manual_seed(seed) # gpu 연산 무작위 고정
  torch.backends.cudnn.deterministic = True  # cuda 라이브러리에서 Deterministic(결정론적)으로 예측하기 (예측에 대한 불확실성 제거 )

def read_csv() : 
    args = easydict.EasyDict()
    # path info
    args.default_path = "data/"
    args.all_csv = args.default_path + "Telco_customer.csv"
    ori_all = pd.read_csv(args.all_csv)
    return ori_all



###################################### Data Cleaning #########################################

# yes, no인 범주형 컬럼을 0과 1로 변환하는 함수
def binary_categorical_to_numeric(df) :
  binary_categorical_columns = ['Partner', 'Dependents','PaperlessBilling']
  for column in binary_categorical_columns :
    df[column] = df[column].apply(lambda x : 1 if x == "Yes" else 0 )
  return df

# 컬럼 삭제 함수
def drop_columns(df) :
  drop_columns = ["customerID", "MonthlyCharges","PhoneService", "StreamingTV", "gender"]
  for col in drop_columns :
    df = df.drop(col,axis=1)
  return df

# lightGBM은 str를 인식 못하므로 category로 변환하는 함수 
def str_to_category(df) :
  columns = ['InternetService' ,'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport','StreamingMovies', 'Contract', 'PaymentMethod']
  for col in columns :
    df[col] = df[col].astype("category")
  return df

