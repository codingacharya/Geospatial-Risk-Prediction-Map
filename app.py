import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk

# 1. Page Configuration
st.set_page_config(
    page_title="Advanced Crime Prediction & XAI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Mock Data Generation (Replace with your ML model inference pipeline)
@st.cache_data
def load_crime_data():
    np.random.seed(42)
    n_rows = 1200
    df = pd.DataFrame({
        'incident_id': [f"INC-{1000 + i}" for i in range(n_rows)],
        'lat': np.random.uniform(17.35, 17.45, n_rows),
        'lon': np.random.uniform(78.40, 78.55, n_rows),
        'crime_type': np.random.choice(['Theft', 'Assault', 'Burglary', 'Vandalism'], n_rows),
        'risk_score': np.random.uniform(0.1, 0.99, n_rows),
        'hour': np.random.randint(0, 24, n_rows),
        'day_of_week': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], n_rows),
        'month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], n_rows)
    })
    return df

df = load_crime_data()

# 3. Sidebar Advanced Controls & Filters
st.sidebar.header("🔍 Advanced Control Panel")

# Map Layer Style Toggle
map_style_type = st.sidebar.radio(
    "Map Visualization Mode",
    ["Scatter Plot (Points)", "Hexagon Density Aggregation"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Filters")

selected_crimes = st.sidebar.multiselect(
    "Select Crime Types",
    options=df['crime_type'].unique(),
    default=df['crime_type'].unique()
)

risk_threshold = st.sidebar.slider(
    "Minimum Risk Score Filter",
    min_value=0.0, max_value=1.0, value=0.3, step=0.05
)

selected_days = st.sidebar.multiselect(
    "Filter by Day of Week",
    options=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    default=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
)

# Hour range filter
hour_range = st.sidebar.slider(
    "Filter by Hour Range",
    min_value=0, max_value=23, value=(0, 23)
)

# Apply Filters
filtered_df = df[
    (df['crime_type'].isin(selected_crimes)) &
    (df['risk_score'] >= risk_threshold) &
    (df['day_of_week'].isin(selected_days)) &
    (df['hour'] >= hour_range[0]) &
    (df['hour'] <= hour_range[1])
].copy()

# 4. Main Dashboard Header
st.title("🛡️ Predictive Policing & Explainable AI Dashboard")
st.markdown("Proactive spatial-temporal risk mapping with local auditability and flexible filtering.")

# Top-level metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Filtered Incidents", len(filtered_df))
col2.metric("Average Risk Score", f"{filtered_df['risk_score'].mean():.2f}" if not filtered_df.empty else "0.0")
col3.metric("Critical Hotspots (>0.75)", len(filtered_df[filtered_df['risk_score'] > 0.75]))
col4.metric("Active Model Architecture", "XGBoost + Spatial LSTM")

st.markdown("---")

# 5. Tabs for Layout Organization
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Spatial Hotspot Map", 
    "📈 Temporal Trends & Analytics", 
    "⚖️ Explainable AI (XAI) Audit",
    "📋 Data Explorer & Export"
])

with tab1:
    st.subheader("Geospatial Risk Prediction Map")
    st.markdown("Inspect forecasted risk distributions across coordinates based on active filters.")
    
    if not filtered_df.empty:
        view_state = pdk.ViewState(
            latitude=filtered_df['lat'].mean(),
            longitude=filtered_df['lon'].mean(),
            zoom=12,
            pitch=40,
        )
        
        if map_style_type == "Scatter Plot (Points)":
            filtered_df['color'] = filtered_df['risk_score'].apply(
                lambda x: [255, int(255 * (1 - x)), 0, 160]
            )
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=filtered_df,
                get_position=['lon', 'lat'],
                get_color='color',
                get_radius=120,
                pickable=True,
            )
            r = pdk.Deck(
                layers=[layer], 
                initial_view_state=view_state, 
                tooltip={"text": "Incident ID: {incident_id}\nCrime: {crime_type}\nRisk Score: {risk_score:.2f}\nHour: {hour}:00"}
            )
        else:
            # Hexagon Aggregation Layer for macro density
            layer = pdk.Layer(
                "HexagonLayer",
                data=filtered_df,
                get_position=['lon', 'lat'],
                radius=100,
                elevation_scale=50,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            )
            r = pdk.Deck(
                layers=[layer], 
                initial_view_state=view_state, 
                tooltip={"text": "Cluster Count & Density Metric"}
            )
            
        st.pydeck_chart(r)
    else:
        st.warning("No data available for the selected filter configuration.")

with tab2:
    st.subheader("Temporal & Cyclical Crime Distribution")
    col_a, col_b = st.columns(2)
    
    with col_a:
        if not filtered_df.empty:
            fig_hour = px.histogram(filtered_df, x="hour", nbins=24, title="Incidents by Hour of Day", color="crime_type")
            st.plotly_chart(fig_hour, use_container_width=True)
            
    with col_b:
        if not filtered_df.empty:
            fig_day = px.histogram(filtered_df, x="day_of_week", title="Incidents by Day of Week", color="crime_type",
                                   category_orders={"day_of_week": ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']})
            st.plotly_chart(fig_day, use_container_width=True)

with tab3:
    st.subheader("Model Transparency & Interpretability (SHAP / LIME)")
    st.markdown("Audit global feature influence and run **Local Surrogate Breakdowns (LIME)** for individual incidents.")
    
    # Global SHAP View
    feature_importance = pd.DataFrame({
        'Feature': ['Hour of Day', 'Historical Incident Density', 'Distance to Transit Hub', 'Weekend Indicator', 'Lighting/Visibility Index', 'Socioeconomic Metric'],
        'Importance (SHAP Value)': [0.34, 0.28, 0.15, 0.12, 0.07, 0.04]
    }).sort_values(by='Importance (SHAP Value)', ascending=True)
    
    fig_shap = px.bar(feature_importance, x='Importance (SHAP Value)', y='Feature', orientation='h', 
                      title="Global Feature Impact on Crime Risk Predictions (Mean Absolute SHAP)")
    st.plotly_chart(fig_shap, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔍 Interactive LIME Local Incident Inspector")
    
    if not filtered_df.empty:
        selected_inc = st.selectbox("Select Incident ID to Inspect Local Explanations", filtered_df['incident_id'].unique())
        inc_data = filtered_df[filtered_df['incident_id'] == selected_inc].iloc[0]
        
        col_x1, col_x2 = st.columns(2)
        with col_x1:
            st.write("**Incident Metadata:**")
            st.json({
                "ID": inc_data['incident_id'],
                "Crime Type": inc_data['crime_type'],
                "Risk Score": round(float(inc_data['risk_score']), 2),
                "Hour": f"{inc_data['hour']}:00",
                "Day": inc_data['day_of_week']
            })
        with col_x2:
            # Generate dummy local feature weights mimicking LIME output for this specific choice
            np.random.seed(int(inc_data['incident_id'].split('-')[1]))
            local_weights = pd.DataFrame({
                'Local Feature Catalyst': ['Recent Local Spike', 'Time Window match', 'Proximity to Hotspot', 'Low Ambient Lighting'],
                'Local Contribution Weight': np.random.uniform(-0.3, 0.4, 4)
            })
            fig_lime = px.bar(local_weights, x='Local Contribution Weight', y='Local Feature Catalyst', orientation='h',
                              title=f"LIME Local Explanation for {selected_inc}")
            st.plotly_chart(fig_lime, use_container_width=True)

with tab4:
    st.subheader("Filtered Dataset Explorer & Export")
    st.markdown("View tabular records corresponding to the current filter parameters and download reports.")
    
    if not filtered_df.empty:
        st.dataframe(filtered_df[['incident_id', 'crime_type', 'risk_score', 'hour', 'day_of_week', 'lat', 'lon']], use_container_width=True)
        
        # CSV Export Button
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Crime Report as CSV",
            data=csv_data,
            file_name="crime_prediction_report.csv",
            mime="text/csv",
        )
    else:
        st.warning("No records to display or export.")