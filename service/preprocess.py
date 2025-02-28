import numpy as np
import pandas as pd
from utils import reset_seeds
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours


def __create_custom_features(df):

    df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
    df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
    df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
    df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
    df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
    df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"

    #df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
    #                                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    #                                    'StreamingTV', 'StreamingMovies']]== 1).sum(axis=1)
    df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]
    #df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)
    #df.drop(columns = ['PhoneService', 'gender','StreamingTV','StreamingMovies','MultipleLines','InternetService'],inplace = True)
    return df

def __cleaning_data(df):
    df=df.dropna()
    num_cols = ['tenure','MonthlyCharges','TotalCharges']
    mms = MinMaxScaler() # Normalization
    df['tenure'] = mms.fit_transform(df[['tenure']])
    df['MonthlyCharges'] = mms.fit_transform(df[['MonthlyCharges']])
    df['TotalCharges'] = mms.fit_transform(df[['TotalCharges']])
    df["NEW_AVG_Charges"] = mms.fit_transform(df[["NEW_AVG_Charges"]])
    #df['NEW_AVG_Service_Fee'] = mms.fit_transform(df[['NEW_AVG_Service_Fee']])
    df['NEW_Increase'] = mms.fit_transform(df[['NEW_Increase']])
    return df
def __smote_data(X,y):
    # 4. SMOTE 적용
    #smote = SMOTE(sampling_strategy=0.5, k_neighbors=3)
    #enn = EditedNearestNeighbours(n_neighbors=3)

    # SMOTEENN 파라미터 조정
    sm = SMOTEENN(sampling_strategy=0.5)
    X_smo, y_smo = sm.fit_resample(X, y)
    return X_smo,y_smo

def __encode_data(df):
    #le=LabelEncoder()
    #df['Churn']=le.fit_transform(df['Churn'])


    numeric_data=df.select_dtypes(include=np.number)
    categorical_data=df.select_dtypes(exclude=np.number)

    #for i in categorical_data.columns:
    #    df[i]=le.fit_transform(categorical_data[i])
    columns = df.select_dtypes(include=['category','object']).columns
    df = pd.get_dummies(df, columns=columns, drop_first=False)
    return df



@reset_seeds
def preprocess_dataset(data):
    df=data.dropna()
    le=LabelEncoder()
    df['Churn']=le.fit_transform(df['Churn'])
    df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
    
    #label_encoder = LabelEncoder()
    #or col in columns:
    #    df[col] = label_encoder.fit_transform(df[col])
    
    df = __create_custom_features(df)
    #df = __encode_data(df)
    df = __cleaning_data(df)
    columns = df.select_dtypes(include=['category','object']).columns
    df = pd.get_dummies(df, columns=columns, drop_first=False)
    y = df["Churn"]
    X = df.drop(columns=["Churn"],axis=1)
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y)


    #x_train, y_train = __smote_data(x_train, y_train)



    return x_train,x_test,y_train,y_test