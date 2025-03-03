import streamlit as st
import pandas as pd
import numpy as np
import joblib
from service.preprocess import __cleaning_data, __encode_data
from service.data import load_data

# CSV 파일 로드
df = load_data()

# TotalCharges의 공백을 NaN으로 변환 후, 숫자로 변환
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)  # 중앙값으로 대체

# 저장된 모델 및 스케일러 로드
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")  # ✅ 학습된 스케일러 불러오기

# Streamlit UI 구성
st.title("📞고객 이탈 예측")
st.write("고객 정보를 입력하면 고객의 이탈 확률을 예측할 수 있습니다.")
st.markdown("---")

user_input = {}

# 사용 가능한 피쳐 목록 가져오기
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# TotalCharges를 수치형으로 간주
if "TotalCharges" in categorical_features:
    categorical_features.remove("TotalCharges")
    numerical_features.append("TotalCharges")

# SeniorCitizen을 범주형으로 간주
if "SeniorCitizen" in numerical_features:
    numerical_features.remove("SeniorCitizen")
    categorical_features.append("SeniorCitizen")

# UI에서 제외할 피쳐 정의
hidden_features = ["customerID", "Churn"]

# 슬라이더 3개를 한 줄에 배치
col1, col2, col3 = st.columns(3)
for i, feature in enumerate(numerical_features):
    if feature not in hidden_features:
        min_val = int(df[feature].min())
        max_val = int(df[feature].max())
        avg_val = int(df[feature].mean())
        if i % 3 == 0:
            user_input[feature] = col1.slider(feature, min_val, max_val, avg_val)
        elif i % 3 == 1:
            user_input[feature] = col2.slider(feature, min_val, max_val, avg_val)
        else:
            user_input[feature] = col3.slider(feature, min_val, max_val, avg_val)

st.markdown("---")

# 셀렉트박스를 한 줄에 4개씩 배치 (SeniorCitizen 포함)
categorical_columns = [feature for feature in categorical_features if feature not in hidden_features]
num_cols = 4
rows = [categorical_columns[i : i + num_cols] for i in range(0, len(categorical_columns), num_cols)]

for row in rows:
    cols = st.columns(len(row))
    for i, feature in enumerate(row):
        unique_values = [0, 1] if feature == "SeniorCitizen" else df[feature].dropna().unique().tolist()
        user_input[feature] = cols[i].selectbox(feature, unique_values)

# 예측 버튼 추가
if st.button("예측하기"):
    # 입력값을 데이터프레임으로 변환
    input_df = pd.DataFrame([user_input])

    # 원본 데이터에 hidden_features 추가 (전처리에 필요할 경우 기본값 설정)
    for feature in hidden_features:
        if feature in df.columns:
            input_df[feature] = df[feature].mode()[0]  # 가장 많이 등장하는 값 사용

    # 1️⃣ 데이터 클리닝 수행
    input_df = __cleaning_data(input_df)

    # 2️⃣ 범주형 데이터 인코딩
    input_df = __encode_data(input_df)

    # 3️⃣ 수치형 변수 스케일링 (저장된 스케일러를 사용하여 transform)
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_df[numeric_features] = scaler.transform(input_df[numeric_features]).astype(float)  # ✅ 학습된 스케일러 적용

    # 모델이 학습한 피쳐 리스트 가져오기
    model_features = model.feature_names_in_

    # 모델이 학습한 컬럼에 없는 컬럼은 제거
    input_df = input_df[[col for col in model_features if col in input_df.columns]]

    # 학습 데이터에는 있었지만, 예측 데이터에는 없는 컬럼을 0으로 채우기
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # 피쳐 순서를 모델 학습 시 사용한 순서와 동일하게 정렬
    input_df = input_df[model_features]

    # 4️⃣ 모델 예측 수행
    prediction = model.predict_proba(input_df)[0][1]  # 이탈 확률

    # 5️⃣ 예측 결과 출력
    st.metric(label="예상 고객 이탈율", value=f"{prediction * 100:.2f}%") 