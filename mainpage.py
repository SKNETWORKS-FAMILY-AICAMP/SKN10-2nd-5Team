import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import lightgbm as lgb
import joblib
from service.preprocess_dl import *
from service.process_dl import * 

# ✅ 전처리 모듈 가져오기
from service.preprocess import *

# 한글 폰트 설정 (Windows의 경우)
font_path = 'C:/Windows/Fonts/malgun.ttf'  # 윈도우에서는 'malgun.ttf' 폰트를 사용
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

st.set_page_config(page_title="통신사 고객 이탈 예측 서비스", page_icon="📈")
st.title("📱 통신사 고객 이탈 예측 서비스")
st.write("고객 정보를 입력하면 고객의 이탈 확률을 예측할 수 있습니다.")
st.divider()

# CSV 파일 로드
df = read_csv()

# ✅ TotalCharges의 공백을 NaN으로 변환 후, 숫자로 변환
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)  # 중앙값으로 대체

# 사용 가능한 피처 목록 가져오기
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

if "TotalCharges" in categorical_features:
    categorical_features.remove("TotalCharges")
    numerical_features.append("TotalCharges")

if "SeniorCitizen" in numerical_features:
    numerical_features.remove("SeniorCitizen")
    categorical_features.append("SeniorCitizen")

hidden_features = ["customerID", "Churn"]

# UI에서 보이지 않도록 필터링된 피처 목록
visible_numerical_features = [f for f in numerical_features if f not in hidden_features]
visible_categorical_features = [f for f in categorical_features if f not in hidden_features]

# 사용자 입력 받기
user_input = {}

col1, col2, col3 = st.columns(3)
for i, feature in enumerate(visible_numerical_features):
    min_val = int(df[feature].min())
    max_val = int(df[feature].max())
    avg_val = int(df[feature].mean())
    if i % 3 == 0:
        user_input[feature] = col1.slider(feature, min_val, max_val, avg_val)
    elif i % 3 == 1:
        user_input[feature] = col2.number_input(feature, min_val, max_val, avg_val)
    else:
        user_input[feature] = col3.number_input(feature, min_val, max_val, avg_val)

st.markdown("---")

num_cols = 4
rows = [visible_categorical_features[i : i + num_cols] for i in range(0, len(visible_categorical_features), num_cols)]

for row in rows:
    cols = st.columns(len(row))
    for i, feature in enumerate(row):
        unique_values = [0, 1] if feature == "SeniorCitizen" else df[feature].dropna().unique().tolist()
        user_input[feature] = cols[i].selectbox(feature, unique_values)

def preprocess_ml(input_df):
    # 이진 범주형을 0 또는 1로 변환
    data = binary_categorical_to_numeric(input_df) 

    # Churn과 TotalCharges를 전처리함.
    data['Churn'] = data['Churn'].apply(lambda x : 1 if x == "Yes" else 0 )
    data['TotalCharges'] = data['TotalCharges'].replace(" ", "0")  # 공백을 '0'으로 변환
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])  # 숫자로 변환

    # 컬럼 추가
    data['notSecurityBackup'] = data.apply(lambda x : 1 if x['OnlineBackup'] == "No" and x['OnlineSecurity'] == "No" else 0, axis=1) # 보안, 백업 서비스를 사용 안하면 1
    data['isAlone'] = data.apply(lambda x : 1 if x['Partner'] == 0 and x['Dependents'] == 0 else 0, axis=1) # 혼자인지 여부
    data['notTechSupport'] = data.apply(lambda x : 1 if x['TechSupport'] == "No" and x['Contract'] == "Month-to-month" else 0, axis=1) # 기술지원 x, 계약기간 짧으면 1
    data["new_avg_charges"] = data["TotalCharges"] / (data["tenure"] + 1)
    data["new_increase"] = data["new_avg_charges"] / data["MonthlyCharges"]

    # 컬럼 삭제
    data = drop_columns(data)
    data.drop(columns=["Churn"], axis=1, inplace=True)

    # str 컬럼을 category로 변환
    data = str_to_category(data)

    return data

def preprocess_dl(input_df):

    df = input_df
    df = create_custom_features(df)
    df = cleaning_data(df) 
    df = encode_data(df)

    return df

# 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드 함수
@st.cache_resource
def load_model(model_stat: str):
    if model_stat == "ML":
        return joblib.load("model/lightgbm_model.pkl")
    elif model_stat == "DL":
        input_size = 24  # 현재 데이터 크기에 맞춰 조정
        output_size = 2  # 이진 분류 모델이라면 2
        hidden_size= 32

        model = MultiModel(input_size=input_size, out_size=output_size, hidden_size=hidden_size)
        model.load_state_dict(torch.load("model/best_model.pth", map_location=device))  
        model.eval()
        return model
    else:
        return None

# 모델 선택하기
st.divider()
st.markdown("<h3 style='margin-bottom: -50px;'>모델 선택</h3>", unsafe_allow_html=True)
model_stat = st.radio('', options=['ML', 'DL'], index=0, horizontal=True)

# 모델 불러오기

if st.button("예측하기"):
    input_df = pd.DataFrame([user_input])

    # 전처리 및 예측 
    if model_stat == "ML":
        # ✅ UI에서 숨겼던 customerID와 Churn을 임시 추가
        input_df["customerID"] = "0000-AAAAA"  # 임의의 ID 값S
        input_df["Churn"] = 0  # 전처리 과정에서 필요하므로 임시 추가
        
        data = preprocess_ml(input_df)

        model = load_model(model_stat)
        
        pred_prob = model.predict_proba(data)[:,1]
        pred_class = (pred_prob >= 0.5).astype(int)
        
        st.markdown(f"### 고객 이탈 확률: {int(pred_prob[0]*100)}%")

    elif model_stat == "DL":
        data = preprocess_dl(input_df)

        model = load_model(model_stat)

        # 🔹 전처리 후 feature 개수 확인
        print(f"전처리된 데이터 shape: {data.shape}")  # (1, feature 개수) 확인
        data_tensor = torch.tensor(data.values, dtype=torch.float32).to(device)

        # 모델 예측 수행
        with torch.no_grad():
            outputs = model(data_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            churn_prob = probabilities[:, 1] * 100

        st.markdown(f"### 예측 결과: {churn_prob.cpu().numpy()[0]:.2f}%")