import numpy as np
import pandas as pd
from service.utils import reset_seeds
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def __encode_data(df):
    df['TotalCharges'].replace(' ', 0, inplace=True)
    df['TotalCharges'] = df['TotalCharges'].astype(float)

    encoder = LabelEncoder()
    columns = df.select_dtypes(exclude=np.number).columns
    for col in columns:
        df[col] = df[col].str.lower()
        df[col] = encoder.fit_transform(df[col])
    
    return df

def __scale_data(X_train, X_test):
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

@reset_seeds
def preprocess_dataset(data):
    data.drop(['customerID', 'gender', 'PhoneService', 'InternetService'], axis=1, inplace=True)
    
    data = __encode_data(data)

    X = data.drop(['Churn'], axis=1)
    y = data['Churn']
    keys = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

    X_train, X_test = __scale_data(X_train, X_test)

    return X_train, X_test, y_train, y_test