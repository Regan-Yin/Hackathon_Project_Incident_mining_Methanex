import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def load_epssc_data():
    """Loads and merges the historical safety datasets."""
    events = pd.read_csv("data/events_clean.csv")
    actions = pd.read_csv("data/actions_clean.csv")
    clusters = pd.read_csv("data/case_cluster_map.csv")
    
    # Merge cluster names into the main events dataframe
    events = events.merge(clusters[['event_id', 'cluster_name']], on='event_id', how='left')
    return events, actions

def calculate_kpis(filtered_df):
    """Calculates top-level metrics for the executive KPI strip."""
    total = len(filtered_df)
    if total == 0:
        return {"total": 0, "inc_pct": 0, "nm_pct": 0, "conversion": 0, "high_risk": 0, "major_serious": 0, "missing_lessons": 0}
        
    incidents = len(filtered_df[filtered_df['category_type'] == 'Incident'])
    near_misses = len(filtered_df[filtered_df['category_type'] == 'Near Miss'])
    
    return {
        "total": total,
        "inc_pct": (incidents / total) * 100,
        "nm_pct": (near_misses / total) * 100,
        "conversion": (incidents / (incidents + near_misses) * 100) if (incidents + near_misses) else 0,
        "high_risk": (len(filtered_df[filtered_df['risk_level'] == 'High']) / total) * 100,
        "major_serious": (len(filtered_df[filtered_df['severity'].isin(['Major', 'Serious'])]) / total) * 100,
        "missing_lessons": (filtered_df['lessons'].isna().sum() / total) * 100
    }

def plot_reported_cases_trend(df):
    """Graph 1: Line chart of cases over time."""
    trend_df = df.groupby(['year', 'category_type']).size().reset_index(name='count')
    fig = px.line(trend_df, x='year', y='count', color='category_type', markers=True, 
                  color_discrete_sequence=["#0033A0", "#78BE20", "#FFA500"])
    fig.add_vline(x=2024, line_dash="dash", line_color="red", annotation_text="Dataset Jump")
    fig.update_layout(title="Reported Cases Over Time", xaxis_title="Year", yaxis_title="Count")
    return fig

def plot_cluster_priority_heatmap(df):
    """Graph 2: Heatmap showing cluster priorities over the years."""
    heat_df = df.groupby(['cluster_name', 'year']).size().reset_index(name='cases')
    fig = go.Figure(data=go.Heatmap(
        z=heat_df['cases'], x=heat_df['year'], y=heat_df['cluster_name'], colorscale='Blues'
    ))
    fig.update_layout(title="Cluster Priority Heatmap", xaxis_title="Year", yaxis_title="Cluster")
    return fig

def plot_frequency_vs_incidentization(df):
    """Graph 3: Bubble quadrant chart mapping lagging vs leading indicators."""
    bubble_df = df.groupby('cluster_name').agg(
        n_cases=('event_id', 'count'),
        incidents=('category_type', lambda x: (x == 'Incident').sum())
    ).reset_index()
    bubble_df['incident_rate'] = bubble_df['incidents'] / bubble_df['n_cases']
    
    fig = px.scatter(bubble_df, x='n_cases', y='incident_rate', text='cluster_name', size='n_cases', 
                     color='incident_rate', color_continuous_scale="Reds")
    fig.update_traces(textposition='top center')
    fig.update_layout(title="Cluster Portfolio: Frequency vs Incidentization", xaxis_title="Frequency (n_cases)", yaxis_title="Incident Rate")
    return fig