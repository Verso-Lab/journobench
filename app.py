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
    "claude-3-5-sonnet-20241022": "Anthropic Claude Sonnet 3.5",
    "gemini-2.5-flash-preview-04-17": "Google Gemini Flash 2.5 Preview",
    "gemini-2.5-pro-preview-03-25": "Google Gemini Pro 2.5 Preview (R)",
    "gpt-4o": "OpenAI GPT-4o",
    "o4-mini-2025-04-16": "OpenAI o4-mini (R)",
    "gpt-4.1": "OpenAI GPT-4.1"
}

# Add language display names
LANGUAGE_DISPLAY_NAMES = {
    "en": "English",
    "de": "Deutsch"
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
@st.cache_data(ttl="1h") # Restore caching
def get_evaluation_data(_db_client): # Reverted parameter name
    """Fetch evaluation data from Firestore, transform the nested structure, 
       and ensure key columns exist. Now fetches from generation_logs.
    """
    if not _db_client:
        st.error("Firestore client is not valid in get_evaluation_data.")
        return pd.DataFrame() # Return empty DataFrame
        
    evaluations_ref = _db_client.collection("generation_logs") 
    docs = evaluations_ref.stream() 
    
    processed_data = [] 
    docs_list = list(docs) 

    for doc in docs_list:
        parent_data = doc.to_dict()
        doc_id = doc.id

        # --- Updated Extraction Logic --- 
        model_id = parent_data.get('modelId')
        task_id = parent_data.get('task') # Use 'task' field
        language = parent_data.get('language')
        
        # Get evaluation timestamp from nested structure
        eval_data = parent_data.get('evaluation', {}) # Get evaluation dict, default to empty
        eval_ts = pd.to_datetime(eval_data.get('lastEvaluationTimestamp'), errors='coerce')

        # Get generation timestamp directly
        gen_ts = pd.to_datetime(parent_data.get('timestamp'), errors='coerce')

        # Get parent generation ID
        gen_id = parent_data.get('generationId')
        # --- End Updated Extraction Logic ---
        
        # Iterate through nested outputs
        outputs = parent_data.get('outputs', {}) 
        if isinstance(outputs, dict):
            for output_key, output_data in outputs.items():
                if isinstance(output_data, dict):
                    record = {
                        'id': f"{doc_id}_{output_key}",
                        'modelId': model_id,
                        'taskId': task_id, # Use extracted task_id
                        'language': language, # Use extracted language
                        'score': output_data.get('score'), # Score is inside output_data
                        'fullOutputText': output_data.get('text'), # Text is inside output_data
                        'evaluationTimestamp': eval_ts, # Use extracted eval_ts
                        'generationTimestamp': gen_ts, # Use extracted gen_ts
                        'parent_generationId': gen_id, 
                        'parent_doc_id': doc_id 
                    }
                    processed_data.append(record)
        else:
            # Handle cases where 'outputs' is not a dict (optional logging)
            # print(f"DEBUG: Skipping doc {doc_id} because 'outputs' is not a dictionary: {outputs}")
            pass 

    if not processed_data:
        return pd.DataFrame() # Return empty if no data processed

    df = pd.DataFrame(processed_data)

    # --- Ensure essential columns exist and have correct types --- 
    # These columns are now populated during the transformation loop,
    # but we still check and handle types carefully.
    required_columns = {
        'taskId': 'Unknown', 
        'score': pd.NA,
        'evaluationTimestamp': pd.NaT,
        'generationTimestamp': pd.NaT,
        'modelId': 'Unknown',
        'language': 'Unknown',
        'fullOutputText': pd.NA
    }

    for col, default_value in required_columns.items():
        if col not in df.columns:
            # This case is less likely now but good for safety
            df[col] = default_value 
            st.warning(f"Column '{col}' missing after transformation, added with default values.")

    # Convert timestamps (safe even if column was just added as NaT)
    # Conversion now happens earlier, but we ensure dtype here
    df['evaluationTimestamp'] = pd.to_datetime(df['evaluationTimestamp'], errors='coerce')
    df['generationTimestamp'] = pd.to_datetime(df['generationTimestamp'], errors='coerce')
    
    # Convert score (safe even if column was just added as NA)
    df['score'] = pd.to_numeric(df['score'], errors='coerce')

    # Ensure timezone is removed if present (consistency)
    if pd.api.types.is_datetime64_any_dtype(df['evaluationTimestamp']):
        if df['evaluationTimestamp'].dt.tz is not None:
            df['evaluationTimestamp'] = df['evaluationTimestamp'].dt.tz_convert(None)
    if pd.api.types.is_datetime64_any_dtype(df['generationTimestamp']): 
        if df['generationTimestamp'].dt.tz is not None:
             df['generationTimestamp'] = df['generationTimestamp'].dt.tz_convert(None)
             
    # Add display name 
    if 'modelId' in df.columns: 
        df['modelDisplayName'] = df['modelId'].map(MODEL_DISPLAY_NAMES).fillna(df['modelId'])
    else: 
         df['modelDisplayName'] = 'Unknown' 
         
    return df

# --- UI Display Functions ---
def display_stats(df):
    """Displays key evaluation statistics in columns."""
    st.header("📊 Evaluation Statistics")

    # Calculate metrics
    num_total_generations = len(df)
    evaluated_df = df.dropna(subset=['score'])
    num_evaluated = len(evaluated_df)
    
    # Calculate evaluation rate, handle division by zero
    evaluation_rate = (num_evaluated / num_total_generations * 100) if num_total_generations > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Generations", num_total_generations)
    with col2:
        st.metric("Evaluated Entries", num_evaluated)
    with col3:
        st.metric("Evaluation Rate", f"{evaluation_rate:.1f}%")
    with col4:
        avg_score = evaluated_df['score'].mean() # Mean ignores NaN
        st.metric("Average Score", f"{avg_score:.2f}" if pd.notna(avg_score) else "N/A")
    with col5:
        if num_evaluated > 0:
             positive_percentage = (evaluated_df['score'] > 0).sum() / num_evaluated * 100
             st.metric("Positive Ratings %", f"{positive_percentage:.1f}%" if pd.notna(positive_percentage) else "N/A")
        else:
             st.metric("Positive Ratings %", "N/A")

def display_performance_chart(df, selected_task):
    """Displays the main model performance bar chart using display names."""
    st.header("🏆 Model Performance Ranking")

    # Filter out rows without a score before grouping for chart metrics
    evaluated_df = df.dropna(subset=['score'])

    if evaluated_df.empty:
        st.warning("No evaluated data available for the selected filters to display performance chart.")
        return

    # Group by display name now, using the evaluated data
    model_stats = evaluated_df.groupby('modelDisplayName').agg(
        Average_Score=('score', 'mean'),
        Evaluated_Count=('score', 'count') # Count only entries with a score
    ).reset_index().sort_values('Average_Score', ascending=False)
    
    fig = px.bar(
        model_stats, 
        x='modelDisplayName', # Use display name for x-axis
        y='Average_Score',
        color='Average_Score',
        labels={'modelDisplayName': 'Model', 'Average_Score': 'Average Score', 'Evaluated_Count': 'Evaluated Count'}, # Update label
        title=f"Model Average Score Comparison{' for Task: ' + selected_task if selected_task != 'All Tasks' else ''}",
        hover_data=['Evaluated_Count'], # Update hover data key
        color_continuous_scale='RdYlGn' 
    )
    fig.update_layout(xaxis_title="Model", yaxis_title="Average Score")
    st.plotly_chart(fig, use_container_width=True)

def display_performance_details(df):
    """Displays the detailed model performance table with expanded metrics."""
    with st.expander("🔍 View Detailed Model Performance Data"):
        
        if df.empty:
            st.warning("No data available for the selected filters to display performance details.")
            return

        # 1. Calculate Total Generations per model (from original df)
        total_gens = df.groupby('modelDisplayName').size().reset_index(name='Total_Generations')

        # 2. Filter evaluated data
        evaluated_df = df.dropna(subset=['score'])

        if evaluated_df.empty:
            st.info("No evaluated entries for the selected filters. Displaying total generations only.")
            # Display only total generations if no evaluations exist for the filter
            total_gens = total_gens.rename(columns={'modelDisplayName': 'Model', 'Total_Generations': 'Total Generations'})
            st.dataframe(total_gens, use_container_width=True, hide_index=True)
            return

        # 3. Calculate Aggregated Stats from evaluated data
        evaluated_stats = evaluated_df.groupby('modelDisplayName').agg(
            Average_Score=('score', 'mean'),
            Evaluated_Count=('score', 'count')
        ).reset_index()

        # 4. Calculate Upvotes and Downvotes
        upvotes = evaluated_df[evaluated_df['score'] > 0].groupby('modelDisplayName').size().reset_index(name='Upvotes')
        downvotes = evaluated_df[evaluated_df['score'] < 0].groupby('modelDisplayName').size().reset_index(name='Downvotes')

        # 5. Merge all metrics together
        # Start with total generations, then merge evaluated stats
        model_details = total_gens.merge(evaluated_stats, on='modelDisplayName', how='left')
        # Merge upvotes and downvotes
        model_details = model_details.merge(upvotes, on='modelDisplayName', how='left')
        model_details = model_details.merge(downvotes, on='modelDisplayName', how='left')

        # Fill NaN values resulting from merges (e.g., model with generations but no evaluations, or evaluations but no up/downvotes)
        model_details.fillna({
            'Average_Score': pd.NA, # Keep NA for score if no evaluations
            'Evaluated_Count': 0,
            'Upvotes': 0,
            'Downvotes': 0
        }, inplace=True)

        # Convert counts to integers
        for col in ['Total_Generations', 'Evaluated_Count', 'Upvotes', 'Downvotes']:
            model_details[col] = model_details[col].astype(int)

        # 6. Calculate Rates/Percentages (handle division by zero)
        model_details['Evaluation_Rate'] = (
            (model_details['Evaluated_Count'] / model_details['Total_Generations'] * 100)
            if model_details['Total_Generations'].gt(0).all() else 0 
        )
        model_details['Upvote_Percentage'] = (
            (model_details['Upvotes'] / model_details['Evaluated_Count'] * 100)
             if model_details['Evaluated_Count'].gt(0).all() else 0
        )
        model_details['Downvote_Percentage'] = (
             (model_details['Downvotes'] / model_details['Evaluated_Count'] * 100)
             if model_details['Evaluated_Count'].gt(0).all() else 0
        )
        
        # Handle potential division by zero on a per-row basis if the .all() check isn't sufficient (e.g., mixed 0s)
        model_details['Evaluation_Rate'] = model_details.apply(
            lambda row: (row['Evaluated_Count'] / row['Total_Generations'] * 100) if row['Total_Generations'] > 0 else 0, axis=1
        )
        model_details['Upvote_Percentage'] = model_details.apply(
            lambda row: (row['Upvotes'] / row['Evaluated_Count'] * 100) if row['Evaluated_Count'] > 0 else 0, axis=1
        )
        model_details['Downvote_Percentage'] = model_details.apply(
            lambda row: (row['Downvotes'] / row['Evaluated_Count'] * 100) if row['Evaluated_Count'] > 0 else 0, axis=1
        )

        # 7. Format for display
        model_details['Evaluation_Rate'] = model_details['Evaluation_Rate'].round(1).astype(str) + '%'
        model_details['Upvote_Percentage'] = model_details['Upvote_Percentage'].round(1).astype(str) + '%'
        model_details['Downvote_Percentage'] = model_details['Downvote_Percentage'].round(1).astype(str) + '%'
        model_details['Average_Score'] = model_details['Average_Score'].round(2) # Keep as number for sorting, display handles format

        # 8. Select, Rename, and Sort columns for display
        display_columns = [
            'modelDisplayName',
            'Total_Generations',
            'Evaluated_Count',
            'Evaluation_Rate',
            'Upvotes',
            'Downvotes',
            'Upvote_Percentage',
            'Downvote_Percentage',
            'Average_Score'
        ]
        model_details_display = model_details[display_columns].rename(columns={
            'modelDisplayName': 'Model',
            'Total_Generations': 'Total Gens',
            'Evaluated_Count': 'Evaluated',
            'Evaluation_Rate': 'Eval Rate %',
            'Upvotes': '👍', 
            'Downvotes': '👎',
            'Upvote_Percentage': 'Upvote %',
            'Downvote_Percentage': 'Downvote %',
            'Average_Score': 'Avg Score'
        })

        # Sort by Average Score (descending), handle potential NAs in sorting
        model_details_display = model_details_display.sort_values("Avg Score", ascending=False, na_position='last')

        st.dataframe(
            model_details_display,
            use_container_width=True,
            hide_index=True,
             column_config={ # Optional: Refine formatting if needed
                "Avg Score": st.column_config.NumberColumn(format="%.2f"),
            }
        )

def display_recent_evaluations(df):
    """Displays the table of recent evaluations using display names, sorted by generation time."""
    with st.expander("📄 View Recent Entries (Sorted by Generation Time)"):
        # Ensure necessary timestamp columns exist
        if 'generationTimestamp' not in df.columns:
            st.warning("Generation timestamp column missing. Cannot display recent entries.")
            return
            
        # Sort by generation timestamp (most recent first)
        recent_evals = df.sort_values('generationTimestamp', ascending=False)

        # Prepare display dataframe
        display_df = recent_evals.copy()
        
        # Handle potential NaN scores for display
        if 'score' in display_df.columns:
            display_df['score_display'] = display_df['score'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "N/A")
        else:
            display_df['score_display'] = "N/A"

        # Select and order columns for display
        display_cols_ordered = [
            'generationTimestamp', 
            'modelDisplayName', 
            'taskId', 
            'score_display', # Use the display-formatted score
            'fullOutputText', # Use correct column name
            'evaluationTimestamp' # Keep evaluation timestamp as well
        ]
        # Filter to only columns that actually exist in the dataframe
        display_cols_existing = [col for col in display_cols_ordered if col in display_df.columns]
        
        st.dataframe(
            display_df[display_cols_existing].head(20), 
            use_container_width=True,
            hide_index=True,
            column_config={
                "generationTimestamp": st.column_config.DatetimeColumn("Generated", format="YYYY-MM-DD HH:mm"),
                "evaluationTimestamp": st.column_config.DatetimeColumn("Evaluated", format="YYYY-MM-DD HH:mm"),
                "score_display": st.column_config.TextColumn("Score"), # Display score as text
                "modelDisplayName": "Model",
                "taskId": "Task ID", 
                "fullOutputText": "Generated Text" # Use correct column name and update label
            }
        )

# --- Main App Logic ---
st.title("JournoBench 💡")
st.caption("Track and compare AI model performance on journalism tasks using real feedback from journalists.")

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
# Use taskId for generating task filter options
all_tasks = ["All Tasks"] 
if 'taskId' in df.columns:
    all_tasks += sorted(df['taskId'].astype(str).unique().tolist())
else:
    st.warning("'taskId' column not found. Cannot create task filter.")
    # Optionally add a default 'Unknown' task if appropriate for your logic
    # all_tasks += ['Unknown'] 

model_display_name_options = ["All Models"] + sorted(df['modelDisplayName'].unique().tolist())

# Add language filter options using display names
all_languages = ["All Languages"] 
if 'language' in df.columns:
    unique_languages = df['language'].astype(str).unique().tolist()
    # Map codes to display names, keeping original code if no mapping exists
    language_display_options = sorted([
        LANGUAGE_DISPLAY_NAMES.get(lang, lang) for lang in unique_languages
    ])
    all_languages += language_display_options

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

# Date Filtering: Apply the filter only to rows that HAVE an evaluationTimestamp.
# This ensures that rows without an evaluationTimestamp (i.e., unevaluated entries)
# are NOT dropped by the date filter itself.
if start_date and end_date and 'evaluationTimestamp' in filtered_df.columns:
     start_datetime = pd.to_datetime(start_date)
     end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)
     
     # Ensure timestamp column is datetime 
     # Note: Applying this conversion repeatedly might be slightly inefficient 
     # but is safe. Could optimize by doing it once after loading if needed.
     filtered_df['evaluationTimestamp'] = pd.to_datetime(filtered_df['evaluationTimestamp'], errors='coerce')

     # Condition 1: Rows that DO NOT have an evaluation timestamp (NaT).
     # These rows should always be kept regardless of the date range selected.
     no_timestamp_cond = filtered_df['evaluationTimestamp'].isna()
     
     # Condition 2: Rows that DO have an evaluation timestamp AND fall within the selected date range.
     has_timestamp_and_in_range_cond = (
         filtered_df['evaluationTimestamp'].notna() & 
         (filtered_df['evaluationTimestamp'] >= start_datetime) & 
         (filtered_df['evaluationTimestamp'] < end_datetime)
     )

     # Keep rows that satisfy EITHER condition 1 OR condition 2
     filtered_df = filtered_df[no_timestamp_cond | has_timestamp_and_in_range_cond]

# Use taskId for filtering
if selected_task != "All Tasks": 
    if 'taskId' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['taskId'] == selected_task]
    else:
        st.warning("Cannot filter by task: 'taskId' column not found in data.")

if selected_model_display_name != "All Models":
    filtered_df = filtered_df[filtered_df['modelDisplayName'] == selected_model_display_name]

# Apply language filter
if selected_language != "All Languages" and 'language' in filtered_df.columns:
    # Find the language code corresponding to the selected display name
    lang_code_to_filter = selected_language # Default to selected value if no match found
    for code, display_name in LANGUAGE_DISPLAY_NAMES.items():
        if display_name == selected_language:
            lang_code_to_filter = code
            break # Found the code, exit the loop
    filtered_df = filtered_df[filtered_df['language'] == lang_code_to_filter]

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
    except Exception as display_error:
         st.error("🚨 An error occurred while displaying the results.")
         st.error(f"Error details: {display_error}")
         st.error(traceback.format_exc())

# Simple Footer
st.markdown("--- ")
st.caption("JournoBench | Built by Verso") 