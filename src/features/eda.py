from sklearn.decomposition import PCA
import numpy as np

def basic_metrics(df):
    """Calculate basic metrics like mean, median, variance."""
    return df.describe()

def compute_dispersion(df):
    """Compute dispersion metrics like variance and standard deviation."""
    return df.var(), df.std()

def bivariate_analysis(df, x_col, y_col):
    """Explore the relationship between two variables."""
    correlation = df[[x_col, y_col]].corr().iloc[0, 1]
    return correlation

def correlation_matrix(df, cols):
    """Compute the correlation matrix for selected columns."""
    return df[cols].corr()


def pca_analysis(df, n_components=2):
    """Perform PCA on the dataset to reduce dimensions."""
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df)
    explained_variance = pca.explained_variance_ratio_
    
    return pca_result, explained_variance
