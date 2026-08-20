"""
Streamlit Dashboard
Shows flags table, recent run logs, and metrics
"""
import streamlit as st
import pandas as pd
import os
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="Autonomous Workflow Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Autonomous Workflow Engine Dashboard")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Flags", "Logs", "Metrics"])

# Paths
flags_path = "data/out/flags.csv"
logs_path = "logs/run.log"
metrics_path = "logs/metrics.csv"

if page == "Flags":
    st.header("Flagged Transactions")
    
    if os.path.exists(flags_path):
        df = pd.read_csv(flags_path)
        
        if len(df) > 0:
            st.metric("Total Flagged", len(df))
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                min_prob = st.slider("Min Probability", 0.0, 1.0, 0.7, 0.01)
            with col2:
                if 'merchant' in df.columns:
                    merchants = ['All'] + list(df['merchant'].unique())
                    selected_merchant = st.selectbox("Merchant", merchants)
            
            # Filter data
            filtered_df = df[df['probability'] >= min_prob].copy()
            if 'merchant' in df.columns and selected_merchant != 'All':
                filtered_df = filtered_df[filtered_df['merchant'] == selected_merchant]
            
            # Display table
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Amount", f"${filtered_df['amount'].sum():,.2f}")
            with col2:
                st.metric("Avg Amount", f"${filtered_df['amount'].mean():,.2f}")
            with col3:
                st.metric("Avg Probability", f"{filtered_df['probability'].mean():.2%}")
            with col4:
                st.metric("High Risk", len(filtered_df[filtered_df['probability'] > 0.9]))
        else:
            st.info("No flagged transactions found")
    else:
        st.warning(f"Flags file not found at {flags_path}")

elif page == "Logs":
    st.header("Recent Run Logs")
    
    if os.path.exists(logs_path):
        # Read last N lines
        with open(logs_path, 'r') as f:
            lines = f.readlines()
        
        # Show last 100 lines
        recent_lines = lines[-100:] if len(lines) > 100 else lines
        
        st.text_area(
            "Log Output",
            value=''.join(recent_lines),
            height=500,
            disabled=True
        )
        
        # Parse and show structured logs
        st.subheader("Structured Log Entries")
        log_entries = []
        for line in recent_lines:
            if 'action' in line.lower() or 'workflow' in line.lower():
                log_entries.append(line.strip())
        
        if log_entries:
            st.text('\n'.join(log_entries[-20:]))  # Last 20 entries
        else:
            st.info("No structured log entries found")
    else:
        st.warning(f"Log file not found at {logs_path}")

elif page == "Metrics":
    st.header("Model Metrics")
    
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Visualize metrics
        if 'split' in df.columns:
            st.subheader("Metrics Comparison")
            
            metric_cols = [col for col in df.columns if col != 'split']
            
            for metric in metric_cols:
                st.bar_chart(
                    df.set_index('split')[metric],
                    use_container_width=True
                )
    else:
        st.warning(f"Metrics file not found at {metrics_path}")
        st.info("Run the training script to generate metrics")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

