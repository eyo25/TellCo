from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
from mlflow import log_metric, log_param, log_artifact
def build_regression_model(df):
    X = df[['engagement_score', 'experience_score']]
    y = df['satisfaction_score']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train a regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    
    return model, mse





def track_model(model, mse):
    # Start MLflow tracking
    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_artifact("model.pkl")  # Save model artifacts
        
    mlflow.end_run()
