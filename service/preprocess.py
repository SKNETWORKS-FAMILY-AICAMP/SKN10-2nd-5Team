import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from service.utils import reset_seeds
from service.utils import reset_seeds


def __cleaning_data(df):
    """ 데이터 클리닝: 중복 제거, 결측치 처리, 불필요한 컬럼 삭제 """
    df.drop_duplicates(keep="first", inplace=True, ignore_index=True)
    df.dropna(inplace=True)

    # customerID 삭제 
    df.drop(columns=["customerID"], inplace=True)

    # TotalCharges를 수치형으로 변환
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 상관계수가 낮은 컬럼 삭제
    df.drop(columns=["gender", "PhoneService"], inplace=True)

    return df


@reset_seeds  # ✅ 시드 고정
def scale_numeric_features(X_train, X_test, numeric_features, scaler_type="standard"):
    """ 수치형 변수 스케일링 함수 """

    if scaler_type.lower() == "standard":
        scaler = StandardScaler()
    elif scaler_type.lower() == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type.lower() == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("scaler_type must be either 'standard', 'minmax', or 'robust'")

    # X_train과 X_test에 동일한 스케일러 적용
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features]).astype(float)
    X_test[numeric_features] = scaler.transform(X_test[numeric_features]).astype(float)

    # 학습된 스케일러 저장
    joblib.dump(scaler, "models/scaler.pkl")

    return X_train, X_test


@reset_seeds  # ✅ 시드 고정
def __encode_data(df):
    """ 범주형 데이터 인코딩 """
    # 이진 변수(Yes/No)를 0과 1로 변환
    binary_columns = ["Partner", "Dependents", "PaperlessBilling", "Churn"]
    for col in binary_columns:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # 다중 클래스 범주형 변수
    multi_class_columns = [
        "MultipleLines", "InternetService", "OnlineSecurity", 
        "OnlineBackup", "DeviceProtection", "TechSupport", 
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]

    # 레이블 인코딩 적용
    le = LabelEncoder()
    for col in multi_class_columns:
        df[col] = le.fit_transform(df[col])

    return df


@reset_seeds  # ✅ 시드 고정
def preprocess_dataset(df):
    """ 데이터 전처리: 클리닝, 인코딩, 스케일링 """
    df = __cleaning_data(df)
    df = __encode_data(df)

    # 독립 변수(X)와 종속 변수(y) 분리
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 데이터 분리 (학습 70%, 테스트 30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y  # ✅ 시드 고정
    )

    # 수치형 변수 스케일링 및 저장
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    X_train, X_test = scale_numeric_features(X_train, X_test, numeric_features, scaler_type="standard")

    return X_train, X_test, y_train, y_test
