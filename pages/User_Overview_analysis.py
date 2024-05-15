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
  return df.head()



# Function to aggregate user data
def aggregate_user_data(df):
    # Ensure the required columns are present in the dataframe
    required_columns = ['Bearer Id', 'IMSI', 'Dur. (ms)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 
                        'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 
                        'Other DL (Bytes)', 'Other UL (Bytes)']
    df = df[required_columns]

    # Group by user (assuming 'IMSI' is the user identifier)
    user_agg = df.groupby('IMSI').agg(
        num_xDR_sessions=('Bearer Id', 'count'),
        total_session_duration=('Dur. (ms)', 'sum'),
        total_DL_data=('Youtube DL (Bytes)', 'sum'),
        total_UL_data=('Youtube UL (Bytes)', 'sum'),
        total_netflix_DL=('Netflix DL (Bytes)', 'sum'),
        total_netflix_UL=('Netflix UL (Bytes)', 'sum'),
        total_gaming_DL=('Gaming DL (Bytes)', 'sum'),
        total_gaming_UL=('Gaming UL (Bytes)', 'sum'),
        total_other_DL=('Other DL (Bytes)', 'sum'),
        total_other_UL=('Other UL (Bytes)', 'sum')
    )

    # Calculate the total data volume per application
    user_agg['total_Youtube_data'] = user_agg['total_DL_data'] + user_agg['total_UL_data']
    user_agg['total_Netflix_data'] = user_agg['total_netflix_DL'] + user_agg['total_netflix_UL']
    user_agg['total_Gaming_data'] = user_agg['total_gaming_DL'] + user_agg['total_gaming_UL']
    user_agg['total_Other_data'] = user_agg['total_other_DL'] + user_agg['total_other_UL']

    return user_agg


# Function to describe all relevant variables and their data types
def describe_variables(df):
    desc = df.dtypes.reset_index()
    desc.columns = ["Variable", "Data Type"]
    return desc


# Function to segment users into deciles and compute total data per decile class
def segment_users_by_deciles(user_agg):
    user_agg['decile'] = pd.qcut(user_agg['total_session_duration'], 10, labels=False) + 1
    decile_data = user_agg.groupby('decile').agg(
        total_DL_UL=('total_DL_data', 'sum')
    ).reset_index()
    return decile_data

# Function to calculate basic metrics
def calculate_basic_metrics(df):
    metrics = df.describe().transpose()
    return metrics

# Function to compute dispersion parameters
def compute_dispersion_parameters(df):
     numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
     dispersion = numeric_df.describe().transpose()[['mean', 'std', 'min', 'max']]
     dispersion['variance'] = numeric_df.var()
     dispersion['range'] = dispersion['max'] - dispersion['min']
     return dispersion

# Function to plot histograms for each variable using Plotly
def plot_histograms(df):
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            fig = px.histogram(df, x=column, nbins=30, title=f'Histogram of {column}')
            st.plotly_chart(fig)


# Function to explore bivariate relationships using Plotly
def bivariate_analysis(user_agg):
    applications = ['total_Youtube_data', 'total_Netflix_data', 'total_Gaming_data', 'total_Other_data']
    for app in applications:
        fig = px.scatter(user_agg, x=app, y=user_agg['total_DL_data'] + user_agg['total_UL_data'],
                         title=f'Relationship between {app} and Total DL+UL Data')
        st.plotly_chart(fig)

# Function to compute correlation matrix and plot heatmap using Plotly
def compute_correlation_matrix(user_agg):
    correlation_matrix = user_agg[['total_Youtube_data', 'total_Netflix_data', 'total_Gaming_data', 'total_Other_data']].corr()
    fig = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=list(correlation_matrix.columns),
        y=list(correlation_matrix.index),
        annotation_text=correlation_matrix.round(2).values,
        colorscale='Viridis'
    )
    return fig

# Function to perform PCA and plot using Plotly
def perform_pca(user_agg):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(user_agg[['total_Youtube_data', 'total_Netflix_data', 'total_Gaming_data', 'total_Other_data']])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)
    explained_variance = pca.explained_variance_ratio_
    
    pca_df = pd.DataFrame(data=pca_result, columns=['PCA Component 1', 'PCA Component 2'])
    fig = px.scatter(pca_df, x='PCA Component 1', y='PCA Component 2', title='PCA Result')
    
    return fig, explained_variance


# Main function
def main():
    # Set page title
    st.title("Telecom User Behavior Analysis")

    # Load data
    df = load_data()

    # Display dataset
    st.expander("Telecom Dataset")
    st.write(df)

    #clean_data


    # Task 1.1: Aggregate User Behavior
    st.expander("Task 1.1 - Aggregate User Behavior")
    user_agg = aggregate_user_data(df)
    st.write(user_agg)

    # Task 1.2: Exploratory Data Analysis
    st.expander("Task 1.2 - Exploratory Data Analysis")
    st.expander("Describe Variables and Data Types")
    variable_description = describe_variables(df)
    st.write(variable_description)

    st.expander("User Segmentation by Deciles")
    decile_data = segment_users_by_deciles(user_agg)
    st.write(decile_data)


    st.expander("Basic Metrics")
    basic_metrics = calculate_basic_metrics(df)
    st.write(basic_metrics)


    st.expander("Non-Graphical Univariate Analysis")
    dispersion_parameters = compute_dispersion_parameters(df)
    st.write(dispersion_parameters)


    st.expander("Graphical Univariate Analysis")
    plot_histograms(df)


    st.expander("Bivariate Analysis")
    bivariate_analysis(user_agg)

    st.expander("Correlation Analysis")
    correlation_matrix_fig = compute_correlation_matrix(user_agg)
    st.plotly_chart(correlation_matrix_fig)


    st.expander("Dimensionality Reduction using PCA")
    pca_fig, explained_variance = perform_pca(user_agg)
    st.write(f'Explained Variance by Component: {explained_variance}')
    st.plotly_chart(pca_fig)








# Run the app
if __name__ == "__main__":
    main()
