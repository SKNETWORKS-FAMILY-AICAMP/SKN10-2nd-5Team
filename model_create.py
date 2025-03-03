from Files.preprocessing import *
from Files.process import *
reset_seeds()
data = read_csv() # 데이터 불러오기

################################ 데이터 전처리 ###################################################

# 이진 범주형을 0 또는 1로 변환
data = binary_categorical_to_numeric(data) 

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

# str 컬럼을 category로 변환
data = str_to_category(data)

################################ 데이터 분리 ###################################################
x_tr, x_te, y_tr, y_te = dataset_split(data)
print(x_tr.shape, x_te.shape, y_tr.shape, y_te.shape)
print(data.info())

################################# 모델 학습, 예측 ################################################

model = train_model(x_tr, x_te, y_tr, y_te)
model_evaluation(model, x_te, y_te)
model.booster_.save_model("model/lightgbm_model.txt")

# 저장된 모델 불러오기
#loaded_model = lgb.Booster(model_file="lightgbm_model.txt")