# Import the necessary libraries
import streamlit as st
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN

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
  
  # Split the data into features and label
  X = data.drop(label_col, axis=1) #assuming the name of the label column is "Fraud"
  y = data[label_col]
  
  # Separating Numerical and Categorical variables
  num_cols = X.select_dtypes(include=np.number).columns
  cat_cols = X.select_dtypes(exclude=np.number).columns

  # Scale numerical variables
  scaler = MinMaxScaler()
  X[num_cols] = scaler.fit_transform(X[num_cols])

  # Oversampling on numerical data using Random oversampling with replacement
  X_num_resampled, y_num_resampled = resample(X[num_cols], y, 
                                              random_state=42, 
                                              sampling_strategy='auto', 
                                              replace=True)

  # Oversampling on categorical data using ADASYN
  X_cat_resampled, y_cat_resampled = ADASYN().fit_resample(X[cat_cols], y)

  # Join resampled numerical and categorical data
  X_resampled = pd.concat([X_num_resampled, X_cat_resampled], axis=1)
  y_resampled = y_num_resampled

  # Join the synthetic dataframe to the original dataframe
  data_resampled = pd.concat([X_resampled, y_resampled], axis=1)
  
  st.write("Synthetic Data:", data_resampled)
