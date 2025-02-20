import numpy as np
import pandas as pd
from utils import reset_seeds
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
def __cleaning_data(df):

    return df


def __create_custom_features(df):

    return df

def __smote_data(X,y):
    # 4. SMOTE 적용
    sm = SMOTEENN()
    X_smo, y_smo = sm.fit_resample(X, y)
    return X_smo,y_smo

def __encode_data(df):
    le=LabelEncoder()
    df['Churn']=le.fit_transform(df['Churn'])
    df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')


    numeric_data=df.select_dtypes(include=np.number)
    categorical_data=df.select_dtypes(exclude=np.number)

    for i in categorical_data.columns:
        df[i]=le.fit_transform(categorical_data[i])
    return df



@reset_seeds
def preprocess_dataset(data):
    df=data.dropna()
    
    df = __encode_data(df)
    
    
    
    
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


    X_smo, y_smo = __smote_data(x_train, y_train)



    return X_smo,x_test,y_smo,y_test