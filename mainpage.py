import pickle
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.sidebar.title('고객 이탈 여부')

st.title('📱 통신사 고객 이탈 예측 서비스')
st.write('고객의 정보를 입력하세요.')
st.divider()

id = st.text_input('고객 ID')

col1, col2 = st.columns(2)
with col1:
    gender = st.radio('성별', ['Male', 'Female'], horizontal=True)
with col2:
    senior = st.radio('노인 여부', ['Yes', 'No'], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    partner = st.radio('파트너 유무', ['Yes', 'No'], horizontal=True)
with col2:
    dependents = st.radio('부양 가족 유무', ['Yes', 'No'], horizontal=True)

tenure = st.slider('가입 기간', 0, 72, 0)

st.divider()

col1, col2 = st.columns(2)
with col1:
    phone_service = st.radio('전화 서비스', ['Yes', 'No'], horizontal=True)
with col2:
    multiple_lines = st.radio('다중 회선 서비스', ['Yes', 'No', 'No Phone Service'], horizontal=True)

st.divider()

internet_service = st.radio('인터넷 서비스', ['DSL', 'Fiber Optic', 'No'], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    online_security = st.radio('온라인 보안 서비스', ['Yes', 'No', 'No Internet Service'], horizontal=True)
with col2:
    online_backup = st.radio('온라인 백업 서비스', ['Yes', 'No', 'No Internet Service'], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    device_protection = st.radio('기기 보호 서비스', ['Yes', 'No', 'No Internet Service'], horizontal=True)
with col2:
    tech_support = st.radio('기술 지원 서비스', ['Yes', 'No', 'No Internet Service'], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    streaming_tv = st.radio('스트리밍 TV 서비스', ['Yes', 'No', 'No Internet Service'], horizontal=True)
with col2:
    streaming_movie = st.radio('스트리밍 영화 서비스', ['Yes', 'No', 'No Internet Service'], horizontal=True)

st.divider()

contract = st.radio('계약 기간', ['Month to Month', 'One Year', 'Two Year'], horizontal=True)

paperless_biling = st.radio('무서류 청구서 여부', ['Yes', 'No'], horizontal=True)

payment_method = st.radio('결제 수단', ['Electronic Check', 'Mailed Check', 'Bank Transfer (Automatic)', 'Credit Card (Automatic)'], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    monthly_charges = st.number_input('월 청구 금액', min_value=0.0, step=0.01)
with col2:
    total_charges = st.number_input('총 청구 금액', min_value=0.0, step=0.01)

st.divider()

if st.button('예측하기'):
    input_data = {
        'SeniorCitizen': 1 if senior == 'Yes' else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'MultipleLines': multiple_lines,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movie,
        'Contract': contract,
        'PaperlessBilling': paperless_biling,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    label_encoder = LabelEncoder()

    for col in ['Partner', 'Dependents', 'MultipleLines', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod']:
        input_data[col] = label_encoder.fit_transform([input_data[col]])[0]

    with open('model/best_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    input_values = np.array(list(input_data.values())).reshape(1, -1)
    prediction = model.predict(input_values)

    st.sidebar.markdown(f'<h3 style="color: {"red" if prediction[0] == 1 else "blue"}; font-weight: bold;">고객 {"이탈" if prediction[0] == 1 else "유지"}</h3>', unsafe_allow_html=True)

    st.sidebar.divider()

    st.sidebar.subheader('고객 정보')
    st.sidebar.markdown(f'1. **고객 ID** - {id}')
    st.sidebar.markdown(f'2. **성별** - {gender}')
    st.sidebar.markdown(f'3. **노인 여부** - {senior}')
    st.sidebar.markdown(f'4. **파트너 유무** - {partner}')
    st.sidebar.markdown(f'5. **부양 가족 유무** - {dependents}')
    st.sidebar.markdown(f'6. **가입 기간** - {tenure}')
    st.sidebar.markdown(f'7. **전화 서비스** - {phone_service}')
    st.sidebar.markdown(f'8. **다중 회선 서비스** - {multiple_lines}')
    st.sidebar.markdown(f'9. **인터넷 서비스** - {internet_service}')
    st.sidebar.markdown(f'10. **온라인 보안 서비스** - {online_security}')
    st.sidebar.markdown(f'11. **온라인 백업 서비스** - {online_backup}')
    st.sidebar.markdown(f'12. **기기 보호 서비스** - {device_protection}')
    st.sidebar.markdown(f'13. **기술 지원 서비스** - {tech_support}')
    st.sidebar.markdown(f'14. **스트리밍 TV 서비스** - {streaming_tv}')
    st.sidebar.markdown(f'15. **스트리밍 영화 서비스** - {streaming_movie}')
    st.sidebar.markdown(f'16. **계약 기간** - {contract}')
    st.sidebar.markdown(f'17. **무서류 청구서 여부** - {paperless_biling}')
    st.sidebar.markdown(f'18. **결제 수단** - {payment_method}')
    st.sidebar.markdown(f'19. **월 청구 금액** - {monthly_charges}')
    st.sidebar.markdown(f'20. **총 청구 금액** - {total_charges}')