import plotly.express as px
import streamlit as st
import pandas as pd
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
from src.data.data_preparation import load_data, clean_data, aggregate_experience_metrics,  perform_kmeans
from src.features.engagement_analysis import normalize_metrics, run_kmeans_clustering, compute_cluster_stats, elbow_method,top_engaged_users_per_app
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
# Distribution of average throughput per handset type

# Create the dashboard
def run_experience_dashboard():
    
    # Load and clean data
    df = load_data()
    df = clean_data(df)
    # Aggregate experience metrics
    experience_metrics = aggregate_experience_metrics(df)

    # Perform K-Means clustering (k=3)
    experience_metrics = perform_kmeans(experience_metrics)
    st.title('User Experience Analytics Dashboard')

    # First row: TCP Retransmission & RTT Boxplots
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Distribution of Average Throughput per Handset Type')
        fig_throughput = px.box(experience_metrics, x='handset_type', y='avg_throughput',
                        title='Average Throughput per Handset Type',
                        labels={'avg_throughput': 'Average Throughput (kbps)', 'Handset Type': 'Handset Type'},
                        template='plotly_dark')
        fig_throughput.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig_throughput)

    with col2:
        st.subheader('Distribution of Average TCP Retransmission per Handset Type')
        fig_tcp = px.box(experience_metrics, x='handset_type', y='avg_tcp_retransmission',
                         title='Average TCP Retransmission per Handset Type',
                         labels={'avg_tcp_retransmission': 'Avg TCP Retransmission (Bytes)', 'handset_type': 'Handset Type'},
                         template='plotly_dark')
        fig_tcp.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig_tcp)

    # Second row: K-Means Clustering 3D Scatter Plot
    st.subheader('K-Means Clustering (k=3) - TCP Retransmission, RTT, and Throughput')
    fig_clusters = px.scatter_3d(experience_metrics, x='avg_tcp_retransmission', y='avg_rtt', z='avg_throughput',
                                 color='cluster', symbol='cluster',
                                 title='K-Means Clustering (3 Clusters)',
                                 labels={'avg_tcp_retransmission': 'Avg TCP Retransmission (Bytes)',
                                         'avg_rtt': 'Avg RTT (ms)',
                                         'avg_throughput': 'Avg Throughput (kbps)'},
                                 template='plotly_dark')
    st.plotly_chart(fig_clusters)

    # Third row: Cluster Analysis
    st.subheader('Cluster Summary')
    cluster_summary = experience_metrics.groupby('cluster').agg(
        avg_tcp_retransmission=('avg_tcp_retransmission', 'mean'),
        avg_rtt=('avg_rtt', 'mean'),
        avg_throughput=('avg_throughput', 'mean'),
        count=('MSISDN/Number', 'count')
    ).reset_index()

    st.dataframe(cluster_summary)

    for i, row in cluster_summary.iterrows():
        st.write(f"Cluster {i}:")
        st.write(f"  - Average TCP Retransmission: {row['avg_tcp_retransmission']}")
        st.write(f"  - Average RTT: {row['avg_rtt']}")
        st.write(f"  - Average Throughput: {row['avg_throughput']}")
        st.write(f"  - Number of users: {row['count']}")

# Run the dashboard
if __name__ == '__main__':
    run_experience_dashboard()