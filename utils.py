"""
utils.py
Visualization utilities for the Multi-Modal Disaster Prediction System.
Plotly charts for dashboard.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

THEME = {
    "primary": "#FF5722",
    "primary_dark": "#E64A19",
    "secondary": "#2196F3",
    "accent": "#FFD600",
    "danger": "#FF1744",
    "bg": "#0a0f1a",
    "card": "#111827",
}

SEVERITY_COLORS = {
    "low": "#4CAF50",
    "moderate": "#FF9800",
    "high": "#FF5722",
    "critical": "#D50000",
}

DISASTER_COLORS = {
    "earthquake": "#FF5722",
    "flood": "#2196F3",
    "hurricane": "#9C27B0",
    "wildfire": "#FF9800",
    "tornado": "#00BCD4",
    "tsunami": "#3F51B5",
}


def disaster_distribution_chart(df):
    counts = df["disaster_type"].value_counts()
    colors = [DISASTER_COLORS.get(t, "#666") for t in counts.index]
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, values=counts.values,
        marker=dict(colors=colors),
        hole=0.45,
        textfont=dict(color="#e0e0e0"),
    )])
    fig.update_layout(
        title=dict(text="Disaster Type Distribution", font=dict(color="#fff", size=16)),
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(font=dict(color="#e0e0e0")),
    )
    return fig


def severity_distribution_chart(df):
    order = ["low", "moderate", "high", "critical"]
    counts = df["severity"].value_counts().reindex(order, fill_value=0)
    colors = [SEVERITY_COLORS[s] for s in order]
    fig = go.Figure(data=[go.Bar(
        x=order, y=counts.values,
        marker=dict(color=colors, line=dict(width=0)),
        text=counts.values, textposition="auto",
        textfont=dict(color="#fff"),
    )])
    fig.update_layout(
        title=dict(text="Severity Distribution", font=dict(color="#fff", size=16)),
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Severity Level", color="#8892a4"),
        yaxis=dict(title="Count", color="#8892a4", gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def sensor_distribution_chart(df, column):
    fig = px.histogram(
        df, x=column, nbins=30,
        color="disaster_type",
        color_discrete_map=DISASTER_COLORS,
        labels={column: column.replace("_", " ").title(), "count": "Count"},
    )
    fig.update_layout(
        title=dict(text=f"{column.replace('_', ' ').title()} by Disaster Type", font=dict(color="#fff", size=14)),
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#8892a4"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#8892a4"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(font=dict(color="#e0e0e0", size=10)),
    )
    return fig


def sensor_heatmap_chart(df):
    sensor_cols = [
        "seismic_activity", "ground_vibration", "temperature_c", "humidity_pct",
        "wind_speed_kmh", "air_pressure_hpa", "rainfall_mm", "water_level_m", "visibility_km",
    ]
    corr = df[sensor_cols].corr()
    labels = [c.replace("_", " ").title() for c in sensor_cols]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale="RdYlBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title=dict(text="Sensor Correlation Heatmap", font=dict(color="#fff", size=16)),
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def disaster_by_severity_chart(df):
    ct = pd.crosstab(df["disaster_type"], df["severity"])
    order = ["low", "moderate", "high", "critical"]
    for s in order:
        if s not in ct.columns:
            ct[s] = 0
    ct = ct[order]

    fig = go.Figure()
    for sev in order:
        fig.add_trace(go.Bar(
            name=sev, x=ct.index, y=ct[sev],
            marker=dict(color=SEVERITY_COLORS[sev]),
        ))
    fig.update_layout(
        barmode="stack",
        title=dict(text="Disaster Type by Severity", font=dict(color="#fff", size=16)),
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#8892a4"),
        yaxis=dict(title="Count", color="#8892a4", gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(font=dict(color="#e0e0e0")),
    )
    return fig


def confusion_matrix_chart(cm, labels, title):
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Greens",
        text=cm, texttemplate="%{text}",
        textfont=dict(size=12, color="#fff"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#fff", size=16)),
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Predicted", color="#8892a4"),
        yaxis=dict(title="Actual", color="#8892a4"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def prediction_gauge_chart(confidence, title="Confidence"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number=dict(suffix="%", font=dict(color="#fff", size=28)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#8892a4"),
            bar=dict(color=THEME["primary"]),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 50], color="rgba(255,87,34,0.15)"),
                dict(range=[50, 80], color="rgba(255,87,34,0.25)"),
                dict(range=[80, 100], color="rgba(255,87,34,0.4)"),
            ],
        ),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#fff", size=14)),
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=30, r=30, t=50, b=20),
    )
    return fig


def radar_chart(sensor_data, sensor_stats):
    categories = [k.replace("_", " ").title() for k in sensor_data.keys()]
    values = []
    for k, v in sensor_data.items():
        if k in sensor_stats:
            mean = sensor_stats[k]["mean"]
            std = sensor_stats[k]["std"]
            normalized = (v - mean) / std if std > 0 else 0
            values.append(round(normalized, 2))
        else:
            values.append(0)

    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure(data=go.Scatterpolar(
        r=values, theta=categories,
        fill="toself",
        fillcolor="rgba(255,87,34,0.2)",
        line=dict(color=THEME["primary"], width=2),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-3, 3], color="#8892a4"),
            angularaxis=dict(color="#8892a4"),
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        title=dict(text="Sensor Profile (Z-score)", font=dict(color="#fff", size=14)),
        margin=dict(l=60, r=60, t=50, b=30),
    )
    return fig
