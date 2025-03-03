import pandas as pd

def load_data():

    raw_data = pd.read_csv(r'data\WA_Fn-UseC_-Telco-Customer-Churn.csv')

    return raw_data