import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine
from src.data.data_preparation import load_data, clean_data
# Load dataset
@st.cache


# Main function
def main():
    # Set page title
    st.title("Telecom User Behavior Analysis")

    # Load data
    df = load_data()

    # Display dataset
    st.subheader("Telecom Dataset")
    st.write(df)


# Run the app
if __name__ == "__main__":
    main()