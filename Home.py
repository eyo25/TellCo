import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine
# Load dataset
@st.cache
def load_data():
    # Load telecom dataset
    df = pd.read_csv("telecom_dataset.csv")
    return df

# Main function
def main():
    # Set page title
    st.title("Telecom User Behavior Analysis")

    # Load data
    df = load_data()

    # Display dataset
    st.subheader("Telecom Dataset")
    st.write(df)

    # Task 1.1: Aggregate User Behavior
    st.subheader("Task 1.1 - Aggregate User Behavior")
    # Code for task 1.1 goes here

    # Task 1.2: Exploratory Data Analysis
    st.subheader("Task 1.2 - Exploratory Data Analysis")
    # Code for task 1.2 goes here

# Run the app
if __name__ == "__main__":
    main()