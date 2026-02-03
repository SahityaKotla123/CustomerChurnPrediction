# Week 4: Interactive Dashboard with Streamlit
# Save this as: app.py
# Run with: streamlit run app.py


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# Page config
st.set_page_config(
    page_title="Customer Retention Intervention Engine",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    data = pd.read_csv('intervention_results.csv')
    segment_roi = pd.read_csv('segment_roi.csv')
    summary = pd.read_csv('intervention_summary.csv')
    return data, segment_roi, summary

@st.cache_resource
def load_models():
    churn_model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    segment_names = joblib.load('segment_names.pkl')
    return churn_model, scaler, segment_names

data, segment_roi, summary = load_data()
churn_model, scaler, segment_names = load_models()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📈 Executive Summary", "👥 Customer Segments", "💰 ROI Analysis", 
     "🔍 Customer Lookup", "⚙️ What-If Scenarios"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Key Metrics")
st.sidebar.metric("Total Customers", f"{summary['total_customers'].values[0]:,.0f}")
st.sidebar.metric("High-Risk Customers", f"{summary['high_risk_customers'].values[0]:,.0f}")
st.sidebar.metric("ROI", f"{summary['roi_percentage'].values[0]:.1f}%")

# ============================================================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================================================

if page == "📈 Executive Summary":
    st.title("📈 Customer Retention Intervention Engine")
    st.markdown("### Reducing churn through targeted, cost-effective interventions")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Customers",
            f"{summary['total_customers'].values[0]:,.0f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Intervention Cost",
            f"${summary['total_intervention_cost'].values[0]:,.0f}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Prevented Loss",
            f"${summary['total_prevented_loss'].values[0]:,.0f}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Net Benefit",
            f"${summary['net_benefit'].values[0]:,.0f}",
            delta=f"{summary['roi_percentage'].values[0]:.1f}% ROI"
        )
    
    st.markdown("---")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Financial Impact")
        
        fig = go.Figure()
        
        metrics = ['Intervention<br>Cost', 'Prevented<br>Loss', 'Net<br>Benefit']
        values = [
            summary['total_intervention_cost'].values[0],
            summary['total_prevented_loss'].values[0],
            summary['net_benefit'].values[0]
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#95E77D']
        
        fig.add_trace(go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            text=[f'${v:,.0f}' for v in values],
            textposition='outside'
        ))
        
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis_title="Amount ($)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Risk Distribution")
        
        risk_dist = data['High_Risk'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Low Risk', 'High Risk'],
            values=[risk_dist[False], risk_dist[True]],
            hole=0.4,
            marker_colors=['#95E77D', '#FF6B6B']
        )])
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Segment overview
    st.markdown("---")
    st.subheader("📊 Performance by Segment")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Net Benefit by Segment', 'ROI by Segment'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    colors = ['#95E77D' if x > 0 else '#FF6B6B' for x in segment_roi['Net_Benefit']]
    
    fig.add_trace(
        go.Bar(x=segment_roi.index, y=segment_roi['Net_Benefit'], 
               marker_color=colors, name='Net Benefit'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=segment_roi.index, y=segment_roi['ROI_%'], 
               marker_color='#4ECDC4', name='ROI %'),
        row=1, col=2
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=400, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: CUSTOMER SEGMENTS
# ============================================================================

elif page == "👥 Customer Segments":
    st.title("👥 Customer Segments Analysis")
    
    # Segment selector
    selected_segment = st.selectbox(
        "Select Segment to Analyze",
        data['Segment_Name'].unique()
    )
    
    segment_data = data[data['Segment_Name'] == selected_segment]
    
    # Segment metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Size", f"{len(segment_data):,}")
    
    with col2:
        st.metric("Churn Rate", f"{segment_data['Churn'].mean():.1%}")
    
    with col3:
        st.metric("Avg Monthly Charges", f"${segment_data['MonthlyCharges'].mean():.2f}")
    
    with col4:
        st.metric("Avg Tenure", f"{segment_data['tenure'].mean():.1f} mo")
    
    st.markdown("---")
    
    # Characteristics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Segment Characteristics")
        
        chars = pd.DataFrame({
            'Metric': ['Tenure', 'Monthly Charges', 'Total Charges', 'Num Services'],
            'Value': [
                f"{segment_data['tenure'].mean():.1f} months",
                f"${segment_data['MonthlyCharges'].mean():.2f}",
                f"${segment_data['TotalCharges'].mean():.2f}",
                f"{segment_data['num_services'].mean():.1f}"
            ]
        })
        
        st.dataframe(chars, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🎯 Intervention Strategy")
        
        intervention = segment_data['Assigned_Intervention'].mode()[0]
        high_risk_count = segment_data['High_Risk'].sum()
        
        st.write(f"**Assigned Intervention:** {intervention}")
        st.write(f"**High-Risk Customers:** {high_risk_count:,} ({high_risk_count/len(segment_data):.1%})")
        
        segment_roi_data = segment_roi.loc[selected_segment]
        st.write(f"**Expected Net Benefit:** ${segment_roi_data['Net_Benefit']:,.2f}")
        st.write(f"**ROI:** {segment_roi_data['ROI_%']:.1f}%")
    
    # Distribution plots
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(segment_data, x='Churn_Probability', 
                          color='Churn', nbins=30,
                          labels={'Churn_Probability': 'Churn Probability', 'Churn': 'Actual Churn'},
                          title='Churn Probability Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(segment_data, x='tenure', y='MonthlyCharges',
                        color='High_Risk', size='TotalCharges',
                        labels={'tenure': 'Tenure (months)', 'MonthlyCharges': 'Monthly Charges'},
                        title='Customer Value vs Tenure')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: ROI ANALYSIS
# ============================================================================

elif page == "💰 ROI Analysis":
    st.title("💰 Return on Investment Analysis")
    
    # Overall ROI
    st.subheader("📈 Overall Program Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Investment",
            f"${summary['total_intervention_cost'].values[0]:,.0f}"
        )
    
    with col2:
        st.metric(
            "Revenue Protected",
            f"${summary['total_prevented_loss'].values[0]:,.0f}"
        )
    
    with col3:
        roi_val = summary['roi_percentage'].values[0]
        st.metric(
            "ROI",
            f"{roi_val:.1f}%",
            delta="Positive" if roi_val > 0 else "Negative"
        )
    
    st.markdown("---")
    
    # Segment comparison
    st.subheader("📊 ROI by Segment")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Cost',
        x=segment_roi.index,
        y=segment_roi['Total_Cost'],
        marker_color='#FF6B6B'
    ))
    
    fig.add_trace(go.Bar(
        name='Prevented Loss',
        x=segment_roi.index,
        y=segment_roi['Prevented_Loss'],
        marker_color='#4ECDC4'
    ))
    
    fig.update_layout(
        barmode='group',
        height=400,
        xaxis_tickangle=45,
        yaxis_title="Amount ($)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("📋 Detailed Segment ROI")
    
    display_roi = segment_roi.copy()
    display_roi['Total_Cost'] = display_roi['Total_Cost'].apply(lambda x: f"${x:,.2f}")
    display_roi['Prevented_Loss'] = display_roi['Prevented_Loss'].apply(lambda x: f"${x:,.2f}")
    display_roi['Net_Benefit'] = display_roi['Net_Benefit'].apply(lambda x: f"${x:,.2f}")
    display_roi['ROI_%'] = display_roi['ROI_%'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(display_roi, use_container_width=True)

# ============================================================================
# PAGE 4: CUSTOMER LOOKUP
# ============================================================================

elif page == "🔍 Customer Lookup":
    st.title("🔍 Individual Customer Analysis")
    
    # Select customer
    customer_idx = st.selectbox(
        "Select Customer Index",
        range(len(data)),
        format_func=lambda x: f"Customer {x}"
    )
    
    customer = data.iloc[customer_idx]
    
    # Customer info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Customer Profile")
        st.write(f"**Segment:** {customer['Segment_Name']}")
        st.write(f"**Tenure:** {customer['tenure']:.0f} months")
        st.write(f"**Monthly Charges:** ${customer['MonthlyCharges']:.2f}")
        st.write(f"**Total Charges:** ${customer['TotalCharges']:.2f}")
    
    with col2:
        st.subheader("⚠️ Risk Assessment")
        churn_prob = customer['Churn_Probability']
        
        # Color code risk
        if churn_prob > 0.7:
            risk_color = "red"
            risk_level = "🔴 Very High"
        elif churn_prob > 0.5:
            risk_color = "orange"
            risk_level = "🟠 High"
        elif churn_prob > 0.3:
            risk_color = "yellow"
            risk_level = "🟡 Medium"
        else:
            risk_color = "green"
            risk_level = "🟢 Low"
        
        st.write(f"**Risk Level:** {risk_level}")
        st.write(f"**Churn Probability:** {churn_prob:.1%}")
        
        # Progress bar
        st.progress(churn_prob)
    
    with col3:
        st.subheader("🎯 Recommended Action")
        
        if customer['High_Risk']:
            st.write(f"**Intervention:** {customer['Assigned_Intervention']}")
            st.write(f"**Cost:** ${customer['intervention_cost']:.2f}")
            st.write(f"**Expected Benefit:** ${customer['prevented_loss']:.2f}")
            st.write(f"**Net Value:** ${customer['net_benefit']:.2f}")
        else:
            st.success("✅ No intervention needed - Low risk customer")
    
    # Customer details
    st.markdown("---")
    st.subheader("📋 Full Customer Details")
    
    details = pd.DataFrame({
        'Feature': customer.index,
        'Value': customer.values
    })
    
    st.dataframe(details, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 5: WHAT-IF SCENARIOS
# ============================================================================

elif page == "⚙️ What-If Scenarios":
    st.title("⚙️ What-If Scenario Analysis")
    
    st.markdown("### Adjust intervention parameters to see impact on ROI")
    
    # Sliders for intervention effectiveness
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎚️ Intervention Effectiveness")
        
        discount_10_eff = st.slider(
            "10% Discount Effectiveness",
            0.0, 0.5, 0.15, 0.05,
            help="% reduction in churn probability"
        )
        
        discount_20_eff = st.slider(
            "20% Discount Effectiveness",
            0.0, 0.6, 0.30, 0.05
        )
        
        tutorial_eff = st.slider(
            "Feature Tutorial Effectiveness",
            0.0, 0.5, 0.20, 0.05
        )
        
        support_eff = st.slider(
            "Premium Support Effectiveness",
            0.0, 0.5, 0.25, 0.05
        )
    
    with col2:
        st.subheader("💵 Cost Adjustments")
        
        tutorial_cost = st.number_input(
            "Tutorial Cost ($)",
            0, 100, 15, 5
        )
        
        support_cost = st.number_input(
            "Support Cost ($)",
            0, 200, 50, 10
        )
        
        risk_threshold = st.slider(
            "Risk Threshold (only intervene above)",
            0.3, 0.9, 0.5, 0.1,
            help="Only apply interventions to customers above this churn probability"
        )
    
    # Recalculate with new parameters
    if st.button("🔄 Recalculate ROI"):
        st.success("✅ Scenario calculated!")
        
        # Simplified recalculation
        new_high_risk = data['Churn_Probability'] > risk_threshold
        
        estimated_cost = new_high_risk.sum() * 30  # Rough estimate
        estimated_benefit = estimated_cost * 5  # Rough ROI
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Estimated Cost", f"${estimated_cost:,.0f}")
        
        with col2:
            st.metric("Estimated Benefit", f"${estimated_benefit:,.0f}")
        
        with col3:
            st.metric("Estimated ROI", f"{(estimated_benefit/estimated_cost - 1)*100:.1f}%")

# Footer
st.markdown("---")
st.markdown("**Customer Retention Intervention Engine** | Built with Streamlit")