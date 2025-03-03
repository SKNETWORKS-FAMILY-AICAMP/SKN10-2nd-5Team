import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib import font_manager
import lightgbm as lgb

# 한글 폰트 설정 (Windows의 경우)
font_path = 'C:/Windows/Fonts/malgun.ttf'  # 윈도우에서는 'malgun.ttf' 폰트를 사용
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

#탭 표시 꾸미기
st.set_page_config(page_title="고객 이탈 예측", page_icon="📈")
st.title("고객 이탈 예측 어플리케이션")

# 모델 불러오기기
@st.cache_resource
def load_model():
    return lgb.Booster(model_file="model/lightgbm_model.txt")
model = load_model()

SeniorCitizen = st.selectbox("만 65세 이상이신가요?", ["예", "아니오"] )
Partner = st.selectbox("배우자가 있나요?", ["예", "아니오"])
Dependents = st.selectbox("부양할 가족이 있나요?", ["예", "아니오"] )
tenure = st.slider("서비스를 몇개월동안 사용하셨나요?", min_value=0, max_value=100, value=30)
st.write(SeniorCitizen)
st.write(Partner)
st.write(Dependents)
st.write(tenure)
#'PaperlessBilling'	'TotalCharges'	'notTechSupport','Partner','Dependents','SeniorCitizen','tenure','PaymentMethod',
#'contract','OnlineBackup','TechSupport','Churn','OnlineSecurity','StreamingMovies','DeviceProtection',
#'InternetService','MultipleLines','notSecurityBackup'
