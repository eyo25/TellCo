# TellCo Telecom Data Analysis

## Project Overview

This project involves analyzing customer data from TellCo, a telecom company, to uncover valuable insights about user engagement, experience, and satisfaction. Using the customer’s Call Detail Records (CDR) and xDR (data session) data, we perform exploratory data analysis (EDA), segmentation, and customer satisfaction prediction. The results of the analysis are visualized on a **Streamlit dashboard**.

### Key Features

- **User Overview Analysis**: Identifying top handsets, handset manufacturers, and aggregating user behavior across various applications.
- **User Engagement Analysis**: Tracking user activity through session frequency, duration, and data traffic. Grouping users based on engagement levels using clustering.
- **User Experience Analysis**: Evaluating network parameters like TCP retransmission, round-trip time (RTT), throughput, and handset type.
- **Customer Satisfaction Analysis**: Computing user satisfaction based on engagement and experience scores and performing clustering for further segmentation.
- **Database Storage**: Results are stored in a PostgreSQL database for future retrieval and analysis.

## Project Structure

```bash
tellco-telecom-analysis/
├── .github/                  # CI/CD setup using GitHub Actions
│   └── workflows
│       └── unittests.yml     # Automate unit tests
├── pages/
│   ├── User_Overview.py       # Streamlit dashboard for User Overview Analysis
│   ├── User_Engagement.py     # Streamlit dashboard for User Engagement Analysis
│   ├── Experience_Analytics.py # Streamlit dashboard for User Experience Analysis
│   └── Satisfaction_Analysis.py # Streamlit dashboard for Satisfaction Analysis
├── requirements.txt          # Required Python libraries
├── README.md                 # Project documentation
├── src/
│   ├── data/
│   │   ├── data_preparation.py # Data preparation and cleaning script
│   └── features/
│       └── engagement_analysis.py # User engagement analysis logic
├── tests/                    # Unit tests for data preparation and feature scripts
└── Dockerfile                # Docker configuration for deployment
