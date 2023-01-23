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
With tab_main:
  # Allow the user to upload a file
  file = st.file_uploader("Upload your dataset in csv format", type=["csv"])

  # Or use a sample dataset
  if st.checkbox("Use sample dataset"):
      file_url = 'https://raw.githubusercontent.com/hoyinli1211/SyntheticData/main/sample-synthetic.csv'
      file = pd.read_csv(file_url)

  if file:
      data = pd.read_csv(file)
      st.write("Dataset", data)
      
  num_records = st.number_input("How many additional records would you like to generate?", min_value=1, value=1000)
  
  
