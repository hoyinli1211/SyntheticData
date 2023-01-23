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
  st.subheader("What is Imbalanced Data?")
  st.markdown("Imbalanced data is a common problem in machine learning where the classes in a dataset are not represented equally. In fraud detection, for example, the number of instances of fraud may be significantly less than the number of instances of non-fraud, leading to a biased model that is less effective at detecting fraud. There are several techniques for addressing imbalanced data, including oversampling the minority class, undersampling the majority class, and using techniques such as SMOTE (Synthetic Minority Over-sampling Technique) and ADASYN (Adaptive Synthetic Sampling). The most appropriate method will depend on the specific dataset and the problem being solved. Some reference papers on imbalanced data are:")
  st.markdown("He, H. and Garcia, E. A. (2008). Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, 20(1), pp.1263-1284.")
  st.markdown("Chawla, N. V., Bowyer, K. W., Hall, L. O. and Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, pp.321-357.")
  st.markdown("Han, H., Wang, W. and Mao, B. (2005). Borderline-SMOTE: A new over-sampling method in imbalanced data sets learning. In International Conference on Intelligent Computing, pp.878-887.")
  

def create_OverRandSampling(df, label_col, num_records):
    # Define the oversampling method
    ros = RandomOverSampler(sampling_strategy='auto')
    # Split the data into features and labels
    X, y = df.drop(label_col, axis=1), df[label_col]
    # Apply oversampling
    X_res, y_res = ros.fit_resample(X, y)
    # Create a new dataframe with the synthetic data
    synthetic_df = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res, columns=[label_col])], axis=1)
    # Select a random sample from the synthetic dataframe
    synthetic_df = synthetic_df.sample(num_records, replace=True)
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
