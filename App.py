import streamlit as st
import pandas as pd
from imblearn.over_sampling import SMOTE

#Sidebar
st.sidebar.title("Instructions:")
st.sidebar.markdown("1. ")
st.sidebar.markdown("2. ")
st.sidebar.markdown("3. ")
st.sidebar.markdown("4. ")

def note():
  st.title("Note")
  st.markdown("This is Note 1")
  st.markdown("This is Note 2")
  
#Main Page
st.title("Synthetic Data Generator")
tabs = st.tabs(["Note","Configuration & Synthetic Data Generation"])

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
      
  num_records = st.number_input("How many additional records would you like to generate?", min_value=1, value=1000)
  label_col = st.selectbox("Select label column", data.columns)
  
  smote = SMOTE(random_state=42)
  # Use SMOTE to oversample the minority class & seperate the dataset into features and labels
  X, y = smote.fit_resample(data.drop(label_col, axis=1), data[label_col])
  
  # Create a new dataframe with the oversampled data
  data_resampled = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)

  # Generate the specified number of records
  data_synthetic = data_resampled.sample(num_records, random_state=42)
  st.write("the synthetic data: ", data_synthetic)
  
