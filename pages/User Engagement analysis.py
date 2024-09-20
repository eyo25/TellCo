
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
import streamlit as st
from src.data.data_preparation import load_data, clean_data, aggregate_engagement_metrics
from src.features.engagement_analysis import normalize_metrics, run_kmeans_clustering, compute_cluster_stats, elbow_method,top_engaged_users_per_app
import plotly.express as px

def run_engagement_dashboard():
    # Load and clean data
    df = load_data()
    df = clean_data(df)
    
    # Aggregate engagement metrics before running Elbow Method or other analyses
    engagement_data = aggregate_engagement_metrics(df)

    # Normalize the engagement metrics (optional, based on your approach)
    normalized_data = normalize_metrics(engagement_data)
    # Create two columns for the first section
    col1, col2 = st.columns(2)

    # Section 1: Elbow Method in the first column
    with col1:
        st.header("1. Elbow Method for K-Means Clustering")
        st.write("Use the Elbow Method to find the optimal number of clusters (K).")
        elbow_fig = elbow_method(normalized_data)
        st.plotly_chart(elbow_fig)

    # Section 2: Top 3 Most Used Applications in the second column
    with col2:
        st.header("2. Top 3 Most Used Applications")
        st.write("Display the top 3 most used applications by traffic.")

        # Summing the total traffic per application
        app_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Youtube DL (Bytes)']
        app_traffic = df[app_columns].sum().sort_values(ascending=False).head(3)

        # Bar chart of top 3 applications
        fig = px.bar(x=app_traffic.index, y=app_traffic.values, 
                     labels={'x': 'Application', 'y': 'Total Traffic (Bytes)'}, 
                     title="Top 3 Most Used Applications by Traffic")
        st.plotly_chart(fig)

    # Create a container for the next section
    with st.container():
        st.header("3. Top 10 Most Engaged Users Per Application")
        st.write("View the top 10 users for each application based on traffic (download data).")
        
        # Create 3 columns for the top engaged users per app
        col3, col4, col5 = st.columns(3)
        
        top_users_per_app = top_engaged_users_per_app(df)
        
        # Display top users for Social Media in the first column
        with col3:
            st.subheader("Top 10 users for Social Media")
            st.write(top_users_per_app['Social Media DL (Bytes)'])
        
        # Display top users for Google in the second column
        with col4:
            st.subheader("Top 10 users for Google")
            st.write(top_users_per_app['Google DL (Bytes)'])
        
        # Display top users for YouTube in the third column
        with col5:
            st.subheader("Top 10 users for YouTube")
            st.write(top_users_per_app['Youtube DL (Bytes)'])

# Run the dashboard
if __name__ == '__main__':
    run_engagement_dashboard()




