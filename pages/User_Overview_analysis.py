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
from src.features.eda import basic_metrics, correlation_matrix, pca_analysis
from src.features.visualizations import bar_chart, scatter_plot, heatmap
from src.data.data_preparation import load_data, clean_data, aggregate_user_data

# Define the columns to include in the correlation matrix
app_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
               'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
# Main function
def run_dashboard():
    # Load and clean data
    df = load_data()
    df = clean_data(df)
    aggregated_data = aggregate_user_data(df)
    
    # Set up Streamlit page layout
    st.title('TellCo User Overview Dashboard')

    # Set up tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Top Handsets and Manufacturers", "PCA Analysis", "Correlation Matrix"])

    # Tab 1: Top Handsets and Manufacturers
    with tab1:
        st.header('Top Handsets and Manufacturers')
        
        # Top 10 Handsets
        top_10_handsets = df['Handset Type'].value_counts().head(10)
        st.subheader('Top 10 Handsets Used by Customers')
        st.bar_chart(top_10_handsets)

        # Top 3 Manufacturers
        top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
        st.subheader('Top 3 Handset Manufacturers')
        st.bar_chart(top_3_manufacturers)
    
    # Tab 2: PCA Analysis
    with tab2:
        st.header('Principal Component Analysis (PCA)')
        
        # Define the columns to use for PCA (download data for each app)
        app_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                       'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']

        # Perform PCA and display results
        pca_result, explained_variance = pca_analysis(df[app_columns])
        
        st.subheader('Explained Variance by Principal Components')
        st.write('Explained Variance:', explained_variance)

        # Display PCA scatter plot (if using 2 components)
        pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
        fig = px.scatter(pca_df, x='PC1', y='PC2', title="PCA Scatter Plot (2 Components)")
        st.plotly_chart(fig)
    
    # Tab 3: Correlation Matrix
    with tab3:
        st.header('Correlation Matrix for Application Data')

        # Compute correlation matrix
        correlation_matrix_result = correlation_matrix(df, app_columns)

        # Plot correlation matrix as a heatmap
        corr_fig = heatmap(correlation_matrix_result, app_columns, 'Correlation Matrix for Application Data')
        st.plotly_chart(corr_fig)





# Run the app
if __name__ == "__main__":
    run_dashboard()
