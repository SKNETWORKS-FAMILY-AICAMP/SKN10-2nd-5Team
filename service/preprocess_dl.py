import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from imblearn.combine import SMOTEENN

from sklearn.impute import SimpleImputer  #결측치 자동
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  #scaling
from imblearn.over_sampling import SMOTE #smote
from sklearn.utils.class_weight import compute_class_weight #scaling- weight
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

#from utils import reset_seeds
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

#DL Model
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
#from tensorflow.keras.optimizers import Adam
#from scikeras.wrappers import KerasClassifier


# 랜덤 시드 설정 (데이터셋 준비 전에 실행)
def dl_set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_custom_features(df):
    df = df.copy()  # 원본 데이터프레임 수정 방지

    #  숫자로 변환 (오류 방지)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df["tenure"] = pd.to_numeric(df["tenure"], errors='coerce')
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors='coerce')

    #  NaN 값 처리
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # 신규 피처 생성
    df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
    df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
    df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
    df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
    df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
    df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

    #  Zero-Division 방지: tenure=0인 경우 1로 설정
    df["tenure_fixed"] = df["tenure"].replace(0, 1)

    #  연산 수행 가능하도록 수정 (무한대 발생 방지)
    df["NEW_AVG_Charges"] = df["TotalCharges"] / df["tenure_fixed"]

    # MonthlyCharges가 0이면 1로 보정하여 Zero-Division 방지
    df["MonthlyCharges_fixed"] = df["MonthlyCharges"].replace(0, 1)
    df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges_fixed"]

    return df

def cleaning_data(df):
    df = df.dropna()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'NEW_AVG_Charges', 'NEW_Increase']

    # 무한대 값 처리
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # inf를 NaN으로 변환
    df.fillna(0, inplace=True)  # NaN 값 채우기

    #  MinMaxScaler 적용
    mms = MinMaxScaler()
    df[num_cols] = mms.fit_transform(df[num_cols])

    return df

def encode_data(df):
    df = df.copy()
    #  Churn이 문자열이면 숫자로 변환
    if 'Churn' in df.columns and df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    #  범주형 변수 원-핫 인코딩
    # categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    enc_cols = df.select_dtypes(include=['object']).columns.tolist()
    normal_cols = list(set(df.columns) - set(enc_cols))
    enc = OneHotEncoder()

    tmp = pd.DataFrame(
    enc.fit_transform(df[enc_cols]).toarray(),
    columns = enc.get_feature_names_out()
    )
    df = pd.concat(   
        [df[normal_cols].reset_index(drop=True), tmp.reset_index(drop=True)]
    , axis=1
    )
    

    #df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df



def smote_data(df: pd.DataFrame, target_column: str = 'Churn', sampling_strategy: float = 0.5):
    # 1. X, y 분리
    if target_column in df.columns:
        X = df.drop(columns=[target_column]).values.astype(np.float32)
        y = df[target_column].values.astype(np.int64)
    else:
        raise KeyError(f"{target_column} 컬럼이 데이터프레임에서 사라졌습니다. 전처리 과정을 다시 확인하세요.")

    # 2. 불균형 데이터 처리 (SMOTEENN 적용)
    sm = SMOTEENN(sampling_strategy=sampling_strategy)
    X_processed, y_processed = sm.fit_resample(X, y)
    
    return X_processed, y_processed

