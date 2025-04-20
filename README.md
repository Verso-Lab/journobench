# JournoBench

A Streamlit dashboard for evaluating AI models' performance on news writing tasks based on user feedback data stored in Firestore.

## Setup

1. Make sure you have Python 3.7+ installed
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Make sure your Firebase credentials are properly set up in `.streamlit/secrets.toml`

## Running the App

Run the app with:
```
streamlit run app.py
```

The app will be available at http://localhost:8501

## Features

- View and filter user evaluations of AI model outputs
- Compare performance across different news writing tasks
- Filter by task type (Headlines, Newsletter Writing, etc.)

## Project Structure

- `app.py` - Main Streamlit application
- `.streamlit/secrets.toml` - Firebase credentials (not tracked in git)
- `requirements.txt` - Python dependencies 