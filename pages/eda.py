import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib import font_manager
import lightgbm as lgb
import os
from PIL import Image

# 한글 폰트 설정 (Windows의 경우)
font_path = 'C:/Windows/Fonts/malgun.ttf'  # 윈도우에서는 'malgun.ttf' 폰트를 사용
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

#탭 표시 꾸미기
st.set_page_config(page_title="EDA_결과", page_icon="📈")
st.title("EDA 결과")
st.write(f"현재 실행 디렉토리: {os.getcwd()}")

# streamlit엔 별도로 이미지를 조정하는건 없으니 PIL를 이용해 이미지를 조정정
image1 = Image.open("image/EDA_1.png").resize((1000,600))
image2 = Image.open("image/EDA_2.png").resize((300,300))
# 조정된 이미지 출력
st.image(image1)
st.image(image2)
