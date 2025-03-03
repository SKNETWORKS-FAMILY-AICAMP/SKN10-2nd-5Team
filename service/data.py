import os
import pandas as pd

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    total_data = pd.read_csv(os.path.join(base_path, '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv'))

    return total_data