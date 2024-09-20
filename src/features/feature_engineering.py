import psycopg2
import os
import sys
from sqlalchemy import create_engine
#from src.DB_Connection.connection import PostgresConnection
from dotenv import load_dotenv
from src.utils.utils import missing_values_table, convert_bytes_to_megabytes
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



load_dotenv()
user = os.environ['PG_USER']
password = os.environ['PG_PASSWORD']
host = os.environ['PG_HOST']
port = os.environ['PG_PORT']
database = os.environ['PG_DATABASE']


# src/feature_engineering.py
import pandas as pd

def aggregate_user_data(df):
    aggregated_data = df.groupby('user_id').agg({
        'xdr_sessions': 'sum',
        'session_duration': 'sum',
        'total_dl': 'sum',
        'total_ul': 'sum'
    })
    aggregated_data['total_data_volume'] = aggregated_data['total_dl'] + aggregated_data['total_ul']
    return aggregated_data

# src/eda.py
def describe_variables(df):
    return df.dtypes  # Example function to return variable types

def segment_users_by_deciles(df):
    df['decile'] = pd.qcut(df['session_duration'], 5, labels=False) + 1
    return df.groupby('decile').agg({
        'session_duration': 'mean',
        'total_data_volume': 'sum'
    })

def calculate_basic_metrics(df):
    return df.describe()

def compute_dispersion_parameters(df):
    return df.std(), df.var()

def plot_histograms(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.histplot(df['session_duration'], kde=True)
    plt.show()
