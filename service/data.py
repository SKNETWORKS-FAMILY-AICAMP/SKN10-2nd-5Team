import pandas as pd

def load_data():

    total_data = pd.read_csv('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv').drop(columns=['customerID'])

    return total_data