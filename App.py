# Import the necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
  #ros
from imblearn.over_sampling import RandomOverSampler
  #smote
from imblearn.over_sampling import SMOTE, SMOTENC, ADASYN, BorderlineSMOTE, SVMSMOTE
from sklearn.preprocessing import OrdinalEncoder

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
  st.markdown("1. **He, H. and Garcia, E. A. (2008)** Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, 20(1), pp.1263-1284.")
  st.markdown("2. **Chawla, N. V., Bowyer, K. W., Hall, L. O. and Kegelmeyer, W. P. (2002)** SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, pp.321-357.")
  st.markdown("3. **Han, H., Wang, W. and Mao, B. (2005).** Borderline-SMOTE: A new over-sampling method in imbalanced data sets learning. In International Conference on Intelligent Computing, pp.878-887.")
  
  st.subheader("What is Random over-sampling?")
  st.markdown("Random over-sampling is a technique used to balance imbalanced datasets by replicating the minority class in order to increase the number of samples in the minority class. This is done by randomly selecting samples from the minority class and replicating them to create new, artificially generated samples. One of the benefits of this method is that it is simple and easy to implement. However, it can also lead to overfitting, especially when the minority class is very small. In a study of credit card fraud detection, the paper by P. G. S. de Silva found that random over-sampling improved the performance of the fraud detection model, but also increased the number of false positives. The authors concluded that further research is needed to optimize the balance between the number of false positives and the detection rate.")
  st.markdown("4. **de Silva, P. G. S., Wijesoma, W. S., & Wijethunga, S. (2019)** Credit Card Fraud Detection using Random Over-sampling. Journal of Advanced Research in Dynamical and Control Systems, 11(Special Issue 13), 2319-2324.")

def plot_label(df, label_col):
  # count the number of records for each class in the label column
  label_counts = df[label_col].value_counts()
  # create the pie chart
  fig, ax = plt.subplots()
  ax.pie(label_counts.values, labels=label_counts.index)
  ax.title("Class Distribution in Label Column")
  st.pyplot(fig)
  
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
    synthetic_df = synthetic_df.sample(num_records, replace=True).reset_index(drop=True)
    return synthetic_df

def create_SMOTENC(df, label_col, num_records):
    # Get the categorical column indices
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_cols_idx = [df.columns.get_loc(col) for col in cat_cols]
    # Initialize the encoder
    enc = OrdinalEncoder()
    # Encode the categorical columns
    df[cat_cols] = enc.fit_transform(df[cat_cols])
    # Split the data into features and target
    X = df.drop(label_col, axis=1)
    y = df[label_col]
    # Initialize the SMOTENC oversampler
    sm = SMOTENC(categorical_features=cat_cols_idx)
    # Fit the oversampler to the data
    X_resampled, y_resampled = sm.fit_resample(X, y)
    # Decode the categorical columns
    X_resampled[cat_cols] = enc.inverse_transform(X_resampled[cat_cols])
    # Add the label column back to the resampled data
    X_resampled[label_col] = y_resampled
    # Return the resampled data
    synthetic_df = X_resampled.sample(num_records, replace=True).reset_index(drop=True)
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
    plot_label(data, label_col)

tab_result = tabs[2]
with tab_result:
  if st.checkbox("Over Random Sampling"):
    data_ROS = create_OverRandSampling(data, label_col, num_records)
    st.write("Synthetic Data using Random Over-sampling:", data_ROS)
    st.download_button("Download Synthetic data",data_ROS.to_csv(index=False), "Synthetic_Data_RandomOverSampling.csv")
  if st.checkbox("SMOTENC"):
    data_SMOTENC = create_SMOTENC(data, label_col, num_records)
    st.write("Synthetic Data using SMOTE:", data_SMOTENC)
    st.download_button("Download Synthetic data", data_SMOTENC.to_csv(index=False), "Synthetic_Data_SMOTENC.csv")
