import plotly.express as px

def bar_chart(df, x_col, y_col, title):
    """Create a bar chart."""
    fig = px.bar(df, x=x_col, y=y_col, title=title)
    return fig

def scatter_plot(df, x_col, y_col, title):
    """Create a scatter plot."""
    fig = px.scatter(df, x=x_col, y=y_col, title=title)
    return fig

def heatmap(df, cols, title):
    """Create a heatmap for the correlation matrix."""
    corr_matrix = df[cols].corr()
    fig = px.imshow(corr_matrix, title=title, color_continuous_scale='Blues')
    return fig
