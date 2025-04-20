import streamlit as st
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
import json
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta
import tempfile
import os
import traceback

st.set_page_config(
    page_title="JournoBench - Model Evaluation",
    page_icon="💡",
    layout="wide",
)

# --- Constants ---
MODEL_DISPLAY_NAMES = {
    "claude-3-7-sonnet-20250219": "Anthropic Claude Sonnet 3.7",
    "gemini-2.5-flash-preview-04-17": "Google Gemini Flash 2.5 Preview",
    "gemini-2.5-pro-preview-03-25": "Google Gemini Pro 2.5 Preview (R)",
    "gpt-4o": "OpenAI GPT-4o",
    "o4-mini-2025-04-16": "OpenAI o4-mini (R)"
}

# --- Firestore Initialization ---
@st.cache_resource
def initialize_firestore():
    """Initialize connection to Firestore using temp file for credentials and return db client"""
    # Check if app is already initialized to prevent re-running
    if firebase_admin._apps:
        return firestore.client()

    cred_dict = {
        "type": st.secrets["firestore"]["type"],
        "project_id": st.secrets["firestore"]["project_id"],
        "private_key_id": st.secrets["firestore"]["private_key_id"],
        "private_key": st.secrets["firestore"]["private_key"].replace('\\n', '\n'),
        "client_email": st.secrets["firestore"]["client_email"],
        "client_id": st.secrets["firestore"]["client_id"],
        "auth_uri": st.secrets["firestore"]["auth_uri"],
        "token_uri": st.secrets["firestore"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firestore"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firestore"]["client_x509_cert_url"],
        "universe_domain": st.secrets["firestore"]["universe_domain"]
    }
    
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
            temp.write(json.dumps(cred_dict).encode())
            temp_path = temp.name
        creds = credentials.Certificate(temp_path)
        firebase_admin.initialize_app(creds)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
    
    return firestore.client()

# --- Data Loading ---
@st.cache_data(ttl="1h")
def get_evaluation_data(_db_client): 
    """Fetch evaluation data from Firestore"""
    if not _db_client:
        return [] # Return empty list if client is not valid
        
    evaluations_ref = _db_client.collection("model_evaluations")
    docs = evaluations_ref.stream() # Use stream for potentially large collections
    
    eval_data = []
    for doc in docs:
        data = doc.to_dict()
        data['id'] = doc.id
        
        # Safely convert timestamp
        ts = data.get('evaluationTimestamp')
        if ts:
            try:
                data['evaluationTimestamp'] = pd.to_datetime(ts)
                if data['evaluationTimestamp'].tzinfo:
                   data['evaluationTimestamp'] = data['evaluationTimestamp'].tz_convert(None) 
            except Exception:
                data['evaluationTimestamp'] = pd.NaT # Use NaT for invalid timestamps
        else:
             data['evaluationTimestamp'] = pd.NaT
             
        eval_data.append(data)
    
    # Add display name right after loading
    df = pd.DataFrame(eval_data)
    if not df.empty and 'modelId' in df.columns:
        df['modelDisplayName'] = df['modelId'].map(MODEL_DISPLAY_NAMES).fillna(df['modelId'])
    elif not df.empty:
         df['modelDisplayName'] = 'Unknown' # Handle case where modelId column might be missing
         
    return df # Return DataFrame directly

# --- UI Display Functions ---
def display_stats(df):
    """Displays key evaluation statistics in columns."""
    st.header("📊 Evaluation Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Evaluations", len(df))
    with col2:
        avg_score = df['score'].mean()
        st.metric("Average Score", f"{avg_score:.2f}" if pd.notna(avg_score) else "N/A")
    with col3:
        if len(df) > 0:
             positive_percentage = (df['score'] > 0).sum() / len(df) * 100
             st.metric("Positive Ratings %", f"{positive_percentage:.1f}%" if pd.notna(positive_percentage) else "N/A")
        else:
             st.metric("Positive Ratings %", "N/A")

def display_performance_chart(df, selected_task):
    """Displays the main model performance bar chart using display names."""
    st.header("🏆 Model Performance Ranking")
    # Group by display name now
    model_stats = df.groupby('modelDisplayName').agg(
        Average_Score=('score', 'mean'),
        Total_Evaluations=('id', 'count')
    ).reset_index().sort_values('Average_Score', ascending=False)
    
    fig = px.bar(
        model_stats, 
        x='modelDisplayName', # Use display name for x-axis
        y='Average_Score',
        color='Average_Score',
        labels={'modelDisplayName': 'Model', 'Average_Score': 'Average Score'}, # Update label
        title=f"Model Average Score Comparison{' for Task: ' + selected_task if selected_task != 'All Tasks' else ''}",
        hover_data=['Total_Evaluations'],
        color_continuous_scale='RdYlGn' 
    )
    fig.update_layout(xaxis_title="Model", yaxis_title="Average Score")
    st.plotly_chart(fig, use_container_width=True)

def display_performance_details(df):
    """Displays the detailed model performance table using display names."""
    with st.expander("🔍 View Detailed Model Performance Data"):
        # Group by display name
        model_stats = df.groupby('modelDisplayName').agg(
            Average_Score=('score', 'mean'),
            Total_Evaluations=('id', 'count')
        ).reset_index()
        
        # Need original ID temporarily for positive count calculation if display name isn't unique (it should be here)
        # Let's assume display name is unique enough for grouping positive counts too
        model_positive = df[df['score'] > 0].groupby('modelDisplayName').size().reset_index(name='Positive_Count')
        model_details = model_stats.merge(model_positive, on='modelDisplayName', how='left').fillna(0)
        
        model_details['Positive_Percentage'] = (
            (model_details['Positive_Count'] / model_details['Total_Evaluations'] * 100)
            .replace([float('inf'), -float('inf')], 0) 
            .fillna(0)
        )
        
        model_details['Positive_Percentage'] = model_details['Positive_Percentage'].round(1).astype(str) + '%'
        model_details['Average_Score'] = model_details['Average_Score'].round(2)
        
        st.dataframe(
            model_details[['modelDisplayName', 'Average_Score', 'Total_Evaluations', 'Positive_Percentage']]
            .sort_values("Average_Score", ascending=False),
            use_container_width=True,
            hide_index=True,
             column_config={"modelDisplayName": "Model"} # Set column header
        )

def display_recent_evaluations(df):
    """Displays the table of recent evaluations using display names."""
    with st.expander("📄 View Recent Evaluation Entries"):
        if 'evaluationTimestamp' in df.columns:
            recent_evals = df.sort_values('evaluationTimestamp', ascending=False)
        else: 
            recent_evals = df 
            
        # Include modelDisplayName, remove modelId
        display_cols = [col for col in ['evaluationTimestamp', 'modelDisplayName', 'task', 'score', 'outputSnippet'] if col in recent_evals.columns]
        
        st.dataframe(
            recent_evals[display_cols].head(20), 
            use_container_width=True,
            hide_index=True,
            column_config={
                "evaluationTimestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
                "score": st.column_config.NumberColumn("Score", format="%d"),
                "modelDisplayName": "Model" # Set column header
            }
        )

def display_example_outputs(df):
    """Displays one positive and one negative example output using display names."""
    st.header("📝 Example Outputs")
    
    if 'evaluationTimestamp' in df.columns:
         df = df.sort_values('evaluationTimestamp', ascending=False)
            
    positive_examples = df[df['score'] > 0]
    negative_examples = df[df['score'] < 0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👍 Positive Example")
        if not positive_examples.empty:
            example = positive_examples.iloc[0]
            # Use modelDisplayName
            for field in ['modelDisplayName', 'task', 'inputHeadline', 'outputSnippet']:
                 if field in example and pd.notna(example[field]):
                      display_field = field.replace('modelDisplayName', 'Model').replace('input','Input ').replace('output','Output ').capitalize()
                      st.markdown(f"**{display_field}:** {example[field]}")
        else:
            st.info("No positive examples match the current filters.")

    with col2:
        st.subheader("👎 Negative Example")
        if not negative_examples.empty:
            example = negative_examples.iloc[0]
            # Use modelDisplayName
            for field in ['modelDisplayName', 'task', 'inputHeadline', 'outputSnippet']:
                 if field in example and pd.notna(example[field]):
                      display_field = field.replace('modelDisplayName', 'Model').replace('input','Input ').replace('output','Output ').capitalize()
                      st.markdown(f"**{display_field}:** {example[field]}")
        else:
             st.info("No negative examples match the current filters.")

# --- Main App Logic ---
st.title("JournoBench 💡")
st.caption("AI Journalism Model Evaluation Dashboard")

st.markdown("Welcome to **JournoBench**! This dashboard visualizes performance evaluations of different AI models on common journalism tasks.")

with st.expander("How JournoBench Works"):
    st.markdown("""
    **How it works:**
    *   The data comes from the **Verso Playground** Chrome extension.
    *   Within the extension, users generate text from various AI models (like GPT-4o, Claude, Gemini) for tasks like headline writing, social copy generation, story angle brainstorming, and more.
    *   Users then evaluate the quality of these AI-generated outputs using simple up/down votes directly in their browser.
    *   This dashboard aggregates and analyzes those evaluations, allowing you to see which models perform best overall and for specific tasks based on real-world user feedback.
    
    Use the filters in the sidebar to explore the data!
    """)

db = None
try:
    db = initialize_firestore()
except Exception as e:
    st.error("🚨 Firestore Initialization Failed")
    st.error(f"Error details: {e}")
    st.error("Please double-check your credentials in `.streamlit/secrets.toml`. Ensure the private key format is correct and all fields match your service account JSON.")
    st.stop() 

# --- Sidebar Filters ---
st.sidebar.header("⚙️ Filters")
df = pd.DataFrame() # Initialize empty dataframe
try:
    with st.spinner("🔄 Loading evaluation data..."):
        df = get_evaluation_data(db) # Now returns DataFrame
except Exception as e:
    st.error("🚨 Failed to load data from Firestore.")
    st.error(f"Error details: {e}")
    st.error(traceback.format_exc())
    st.stop()

if df.empty:
    st.warning("⚠️ No evaluation data found in the `model_evaluations` collection or failed to load.")
    st.stop()

# --- Filter Setup ---
all_tasks = ["All Tasks"] + sorted(df['task'].astype(str).unique().tolist())
model_display_name_options = ["All Models"] + sorted(df['modelDisplayName'].unique().tolist())

# Add language filter options
all_languages = ["All Languages"] 
if 'language' in df.columns:
    all_languages += sorted(df['language'].astype(str).unique().tolist())

min_date, max_date, start_date, end_date = None, None, None, None
if 'evaluationTimestamp' in df.columns and not df['evaluationTimestamp'].isnull().all():
    valid_dates = df['evaluationTimestamp'].dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

if min_date and max_date:
    st.sidebar.subheader("🗓️ Date Range")
    start_date = st.sidebar.date_input("Start", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End", max_date, min_value=min_date, max_value=max_date)
else:
    st.sidebar.info("No date data for filtering.")

selected_task = st.sidebar.selectbox("🏷️ Select Task", all_tasks)
selected_model_display_name = st.sidebar.selectbox("🤖 Select Model", model_display_name_options)
# Add language selectbox
selected_language = st.sidebar.selectbox("🌐 Select Language", all_languages)

# --- Apply Filters ---
filtered_df = df.copy()

if start_date and end_date and 'evaluationTimestamp' in filtered_df.columns:
     start_datetime = pd.to_datetime(start_date)
     end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)
     # Ensure timestamp column is datetime before filtering
     filtered_df['evaluationTimestamp'] = pd.to_datetime(filtered_df['evaluationTimestamp'], errors='coerce')
     filtered_df = filtered_df.dropna(subset=['evaluationTimestamp']) # Remove rows where conversion failed
     filtered_df = filtered_df[
         (filtered_df['evaluationTimestamp'] >= start_datetime) & 
         (filtered_df['evaluationTimestamp'] < end_datetime)
     ]

if selected_task != "All Tasks":
    filtered_df = filtered_df[filtered_df['task'] == selected_task]

if selected_model_display_name != "All Models":
    filtered_df = filtered_df[filtered_df['modelDisplayName'] == selected_model_display_name]

# Apply language filter
if selected_language != "All Languages" and 'language' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['language'] == selected_language]

# --- Display Content ---
if filtered_df.empty:
     st.warning("☢️ No data matches the selected filters.")
else:
    try:
        display_stats(filtered_df)
        st.divider()
        # Pass the original selected_task filter value for the title consistency
        display_performance_chart(filtered_df, selected_task)
        st.divider()
        display_performance_details(filtered_df)
        display_recent_evaluations(filtered_df)
        st.divider()
        display_example_outputs(filtered_df)
    except Exception as display_error:
         st.error("🚨 An error occurred while displaying the results.")
         st.error(f"Error details: {display_error}")
         st.error(traceback.format_exc())

# Simple Footer
st.markdown("--- ")
st.caption("JournoBench | Built with Streamlit") 