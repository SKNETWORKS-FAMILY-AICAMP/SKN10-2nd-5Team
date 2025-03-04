import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
from PIL import Image

# 한글 폰트 설정 (Windows의 경우)
font_path = 'C:/Windows/Fonts/malgun.ttf'  # 윈도우에서는 'malgun.ttf' 폰트를 사용
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# 탭 표시 꾸미기
st.set_page_config(page_title="EDA", page_icon="📈")
st.title("EDA")
st.markdown("---")

st.markdown("## **1. 히트맵**")
st.write("각 특성들의 연관도 확인")
st.image("images/EDA_1.png", width=800)

st.markdown("## **2. 카이제곱 검정**")
st.write("- Churn(이탈 여부)과 특성 간의 관련도")
st.write("- gender와 PhoneService가 Churn과 관련이 적음")
st.image("images/EDA_2.png", width=400)

st.markdown("## **3. 막대 그래프**")
st.write("특성에 따른 Churn의 분포")
st.image("images/EDA_3.png", width=800)

st.markdown("## **4. 데이터 분석**")
st.write("**Senior** 고객(1142명) 중 이탈한 노인 고객(276명)이 41%를 차지한다.")
st.image("images/EDA_6.png", width=600)

st.write("이탈한 고객 중 **OnlineSecurity**와 **OnlineBackup** 서비스를 사용하지 않는 고객이 48%를 차지한다.")
st.image("images/EDA_7.png", width=600)

st.write("**InternetService**를 **Fiber optic**으로 사용하는 사람이 제일 돈을 많이 지불한다.")
st.image("images/EDA_8.png", width=600)

st.write("이탈한 고객(1869명)의 대부분이 **Month-to-Month** 계약을 했고, 그 중에서 **TechSupport** 서비스를 사용하지 않는 고객이 1350명이다.")
st.image("images/EDA_9.png", width=800)
