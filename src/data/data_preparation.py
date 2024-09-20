import pandas as pd
import numpy as np
import psycopg2
import os
import sys
from sqlalchemy import create_engine
#from src.DB_Connection.connection import PostgresConnection
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sqlalchemy import create_engine
load_dotenv()
user = os.environ['PG_USER']
password = os.environ['PG_PASSWORD']
host = os.environ['PG_HOST']
port = os.environ['PG_PORT']
database = os.environ['PG_DATABASE']


def data_from_postgres(query):
    # Create the URI
    uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    
    # Create the engine
    try:
        alchemyEngine = create_engine(uri)
        
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        sys.exit(1)
    print("Engine created!")        
    # Connect to PostgreSQL server
    try:
        dbConnection = alchemyEngine.connect()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        sys.exit(1)

    print("Connection established")
    df = pd.read_sql(query, dbConnection)
    # Close connection
    dbConnection.close()

    return df


# Load dataset

def load_data():
    # create query
    query = """ SELECT * FROM public.xdr_data  """
    df = data_from_postgres(query)
    #closing the connection
    return df


def clean_data(df):
    """Clean the dataset by handling missing values and outliers, focusing on numeric columns."""

    # Separate numeric columns from non-numeric
    numeric_cols = df.select_dtypes(include=['number']).columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    
    # Fill missing values in numeric columns with the column mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    #df.fillna({
       # 'Avg RTT DL (ms)': df['Avg RTT DL (ms)'].mean(),
       # 'Avg RTT UL (ms)': df['Avg RTT UL (ms)'].mean(),
       # 'Avg Bearer TP DL (kbps)': df['Avg Bearer TP DL (kbps)'].mean(),
       # 'Avg Bearer TP UL (kbps)': df['Avg Bearer TP UL (kbps)'].mean(),
       # 'TCP DL Retrans. Vol (Bytes)': df['TCP DL Retrans. Vol (Bytes)'].mean(),
       # 'TCP UL Retrans. Vol (Bytes)': df['TCP UL Retrans. Vol (Bytes)'].mean(),
       # 'Handset Type': df['Handset Type'].mode()[0],
    #}, inplace=True)
    return df
    
    # For non-numeric columns, you could fill missing values with a placeholder (or drop them)
    df[non_numeric_cols] = df[non_numeric_cols].dropna()
    
    # Optionally, handle outliers here using the same numeric columns (e.g., IQR method)
    # For example:
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    # Cap outliers by replacing them with the mean
    outliers = (df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))
    df[numeric_cols] = df[numeric_cols].mask(outliers, df[numeric_cols].mean(), axis=1)
    
    return df

def aggregate_user_data(df):
    """Aggregate data per user using 'MSISDN/Number' as the user identifier."""
    
    # Group by user using 'MSISDN/Number'
    aggregated_data = df.groupby('MSISDN/Number').agg(
        num_xdr_sessions=('Bearer Id', 'count'),  # Number of sessions (assuming Bearer Id identifies unique sessions)
        total_duration=('Dur. (ms)', 'sum'),  # Total session duration
        total_download=('Total DL (Bytes)', 'sum'),  # Total download data
        total_upload=('Total UL (Bytes)', 'sum'),  # Total upload data
        social_media_dl=('Social Media DL (Bytes)', 'sum'),  # Social media download
        social_media_ul=('Social Media UL (Bytes)', 'sum'),  # Social media upload
        google_dl=('Google DL (Bytes)', 'sum'),  # Google download
        google_ul=('Google UL (Bytes)', 'sum'),  # Google upload
        youtube_dl=('Youtube DL (Bytes)', 'sum'),  # YouTube download
        youtube_ul=('Youtube UL (Bytes)', 'sum'),  # YouTube upload
        netflix_dl=('Netflix DL (Bytes)', 'sum'),  # Netflix download
        netflix_ul=('Netflix UL (Bytes)', 'sum'),  # Netflix upload
        gaming_dl=('Gaming DL (Bytes)', 'sum'),  # Gaming download
        gaming_ul=('Gaming UL (Bytes)', 'sum'),  # Gaming upload
        other_dl=('Other DL (Bytes)', 'sum'),  # Other download
        other_ul=('Other UL (Bytes)', 'sum')  # Other upload
    ).reset_index()
    
    return aggregated_data

def aggregate_engagement_metrics(df):
    """Aggregate session frequency, session duration, and total traffic per customer."""
    aggregated_data = df.groupby('MSISDN/Number').agg(
        session_frequency=('Bearer Id', 'count'),  # Count of sessions
        total_duration=('Dur. (ms)', 'sum'),  # Total session duration
        total_download=('Total DL (Bytes)', 'sum'),  # Total download traffic
        total_upload=('Total UL (Bytes)', 'sum')  # Total upload traffic
    ).reset_index()
    
    # Calculate total traffic (download + upload)
    aggregated_data['total_traffic'] = aggregated_data['total_download'] + aggregated_data['total_upload']
    
    return aggregated_data

# Function to aggregate experience metrics per customer
def aggregate_experience_metrics(df):
    """Aggregate average experience metrics per customer."""
    aggregated_data = df.groupby('MSISDN/Number').agg(
        avg_tcp_retransmission=('TCP DL Retrans. Vol (Bytes)', 'mean'),
        avg_rtt=('Avg RTT DL (ms)', 'mean'),
        avg_throughput=('Avg Bearer TP DL (kbps)', 'mean'),
        handset_type=('Handset Type', 'first')  # Assuming a customer uses one handset type
    ).reset_index()

    return aggregated_data

def perform_kmeans(df, n_clusters=3):
    # Only handle the numeric columns
    numeric_columns = ['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']
    
    # Fill NaN values in numeric columns with their column mean
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Fill any remaining NaN values with 0 as a fallback
    df[numeric_columns] = df[numeric_columns].fillna(0)

    # Check if there are still NaN values
    if df[numeric_columns].isnull().any().any():
        # Print the columns with NaN values for debugging
        print("Columns with NaN values:", df[numeric_columns].isnull().sum())
        raise ValueError("There are still NaN values in the input features for clustering.")

    # Normalize numeric features
    scaler = StandardScaler()
    features = df[numeric_columns]
    scaled_features = scaler.fit_transform(features)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    
    return df

def calculate_scores(df, engagement_cluster_centers, experience_cluster_centers):
    # Calculate engagement score (distance from least engaged cluster center)
    engagement_scores = pairwise_distances(df[['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']],
                                           [engagement_cluster_centers[0]])  # Assuming cluster 0 is least engaged
    
    # Calculate experience score (distance from worst experience cluster center)
    experience_scores = pairwise_distances(df[['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']],
                                           [experience_cluster_centers[0]])  # Assuming cluster 0 is worst experience
    
    # Store scores in the dataframe
    df['engagement_score'] = engagement_scores
    df['experience_score'] = experience_scores
    
    return df


def calculate_satisfaction_score(df):
    # Satisfaction score is the average of engagement and experience scores
    df['satisfaction_score'] = df[['engagement_score', 'experience_score']].mean(axis=1)
    
    # Top 10 satisfied customers
    top_10_satisfied = df.nlargest(10, 'satisfaction_score')
    
    return df, top_10_satisfied

from sklearn.cluster import KMeans

def kmeans_clustering(df, k=2):
    # Fit K-means on engagement and experience scores
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['satisfaction_cluster'] = kmeans.fit_predict(df[['engagement_score', 'experience_score']])
    
    return df, kmeans

def aggregate_scores_per_cluster(df):
    # Group by clusters and calculate average scores
    cluster_summary = df.groupby('satisfaction_cluster').agg(
        avg_satisfaction=('satisfaction_score', 'mean'),
        avg_experience=('experience_score', 'mean')
    ).reset_index()
    
    return cluster_summary

def export_to_mysql(df):
    # Create a connection to your MySQL database
    engine = create_engine('mysql+pymysql://username:password@localhost/dbname')
    
    # Export dataframe to MySQL
    df.to_sql('user_satisfaction', con=engine, if_exists='replace', index=False)
    
    # Example select query to confirm export
    result = engine.execute("SELECT * FROM user_satisfaction LIMIT 10")
    for row in result:
        print(row)