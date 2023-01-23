# Import the necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
  #ros
from imblearn.over_sampling import RandomOverSampler


#Sidebar
st.sidebar.title("Instructions:")
st.sidebar.markdown("1. Upload the data")
st.sidebar.markdown("2. Choose the number of records required in the synthetic dataset")
st.sidebar.markdown("3. Select the label data field")
st.sidebar.markdown("4. Choose the synthetic data method")

def note():
  st.title("Note")
  st.markdown("This is Note 1")
  st.markdown("This is Note 2")

def create_OverRandSampling(df, label_col, num_records):
    # Define the oversampling method
    ros = RandomOverSampler(sampling_strategy='auto')
    # Split the data into features and labels
    X, y = df.drop(label_col, axis=1), df[label_col]
    # Apply oversampling
    X_res, y_res = ros.fit_resample(X, y, replace=True)
    # Create a new dataframe with the synthetic data
    synthetic_df = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res, columns=[label_col])], axis=1)
    # Select a random sample from the synthetic dataframe
    synthetic_df = synthetic_df.sample(num_records)
    return synthetic_df
 
#Main Page
st.title("Synthetic Data Generator")
tabs = st.tabs(["Note","Upload & Configuration", "Create Synthetic Data"])

tab_note = tabs[0]
with tab_note:
    note()
    
tab_main = tabs[1]
with tab_main:
  # Allow the user to upload a file
  uploaded_file = st.file_uploader("Upload your dataset in csv format", type=["csv"])

  # Or use a sample dataset
  samplecheck = st.checkbox("Use sample dataset")
  if samplecheck:
      file_url = "https://raw.githubusercontent.com/hoyinli1211/SyntheticData/main/sample-synthetic.csv"
      data = pd.read_csv(file_url)
      st.write("Dataset", data)

  if uploaded_file is not None:
      data = pd.read_csv(file)
      st.write("Dataset", data)
      
  if samplecheck or uploaded_file is not None:
    num_records = st.number_input("How many additional records would you like to generate?", min_value=1, value=1000)
    label_col = st.selectbox("Select label column", data.columns)

tab_result = tabs[2]
with tab_result:
  if st.checkbox("Over Random Sampling"):
    data_ROS = create_OverRandSampling(data, label_col, num_records)
    st.write("Synthetic Data using Random Over-sampling:", data_ROS)
    st.download_button("Download Synthetic data",data_ROS.to_csv(index=False), "Synthetic_Data_RandomOverSampling.csv")
