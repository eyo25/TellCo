from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np

def normalize_metrics(df):
    """Normalize session frequency, session duration, and total traffic."""
    scaler = StandardScaler()
    metrics = ['session_frequency', 'total_duration', 'total_traffic']
    df[metrics] = scaler.fit_transform(df[metrics])
    return df

def run_kmeans_clustering(df, k=3):
    """Run K-Means clustering on the normalized data."""
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['session_frequency', 'total_duration', 'total_traffic']])
    return df

def compute_cluster_stats(df):
    """Compute min, max, average, and total for each cluster."""
    metrics = ['session_frequency', 'total_duration', 'total_traffic']
    cluster_stats = df.groupby('cluster')[metrics].agg(
        min_session_freq=('session_frequency', 'min'),
        max_session_freq=('session_frequency', 'max'),
        avg_session_freq=('session_frequency', 'mean'),
        total_session_freq=('session_frequency', 'sum'),
        min_total_duration=('total_duration', 'min'),
        max_total_duration=('total_duration', 'max'),
        avg_total_duration=('total_duration', 'mean'),
        total_duration=('total_duration', 'sum'),
        min_total_traffic=('total_traffic', 'min'),
        max_total_traffic=('total_traffic', 'max'),
        avg_total_traffic=('total_traffic', 'mean'),
        total_traffic=('total_traffic', 'sum')
    ).reset_index()
    return cluster_stats

def elbow_method(df, max_k=10):
    """Use the elbow method to find the optimal number of clusters, using Plotly for visualization."""
    metrics = ['session_frequency', 'total_duration', 'total_traffic']
    X = df[metrics]
    
    # Make sure max_k is not larger than the number of samples
    max_k = min(max_k, len(df))
    
    sse = []
    k_values = list(range(1, max_k + 1))
    
    # Compute SSE for each k
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    
    # Create Plotly figure
    fig = px.line(x=k_values, y=sse, markers=True, 
                  labels={'x': 'Number of clusters (k)', 'y': 'Sum of Squared Distances (SSE)'},
                  title="Elbow Method for Optimal k")
    return fig


def top_engaged_users_per_app(df):
    """Identify the top 10 most engaged users per application."""
    app_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                   'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    
    top_users = {}
    for app in app_columns:
        top_users[app] = df[['MSISDN/Number', app]].nlargest(10, app)
    
    return top_users