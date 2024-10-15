
#streamlit run Breast_Cancer_Analysis_App.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
#Streamlit

import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.impute import SimpleImputer  # Import the imputer

# Dummy dataset loading (replace with actual data)
data = pd.read_csv(r"D:\Data_Science&AI\Spyder\Breast_Cancer_Project\breast_cancer_data.csv")

# Separate features and target
X = data.drop('Class', axis=1)  # Replace 'target' with the actual target column name
y = data['Class']


# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')  # You can change the strategy ('mean', 'median', etc.)
X_imputed = imputer.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# print percentage of missing values in the numerical variables in training set
for col in X_train.columns:
    if X_train[col].isnull().mean()>0:
        print(col,round(X_train[col].isnull().mean(),4))
# Bare_Nuclei 0.0233
# impute missing values in X_train and X_test with respective column median in X_train

for df1 in [X_train, X_test]:
    for col in X_train.columns:
        col_median=X_train[col].median()
        df1[col]= df1[col].fillna(col_median)      
# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model (assume k=5)
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train_scaled, y_train)  # Make sure you're fitting the actual model

# Interactive user input via sliders for prediction
st.write("<h1 style='text-align: left; color: purple;'>Breast Cancer Analysis</h1>", unsafe_allow_html=True)



# Dummy data input for this example
input_data = {
    'Clump_thickness': [st.slider('Clump_thickness', min_value=1, max_value=10, value=1, step=1)],
    'Uniformity_Cell_Size': [st.slider('Uniformity_Cell_Size', min_value=1, max_value=10, value=1, step=1)],
    'Uniformity_Cell_Shape': [st.slider('Uniformity_Cell_Shape', min_value=1, max_value=10, value=1, step=1)],
    'Marginal_Adhesion': [st.slider('Marginal_Adhesion', min_value=1, max_value=10, value=1, step=1)],
    'Single_Epithelial_Cell_Size': [st.slider('Single_Epithelial_Cell_Size', min_value=1, max_value=10, value=1, step=1)],
    'Bare_Nuclei': [st.slider('Bare_Nuclei', min_value=1, max_value=10, value=1, step=1)],
    'Bland_Chromatin': [st.slider('Bland_Chromatin', min_value=1, max_value=10, value=1, step=1)],
    'Normal_Nucleoli': [st.slider('Normal_Nucleoli', min_value=1, max_value=10, value=1, step=1)],
    'Mitoses': [st.slider('Mitoses', min_value=1, max_value=10, value=1, step=1)]
}

#Convert the input into a DataFrame for prediction
input_data_df = pd.DataFrame(input_data)

# Impute any missing values in the input data
input_data_imputed = imputer.transform(input_data_df)

# Ensure all values are numeric (this is an extra check)
input_data_df = pd.DataFrame(input_data_imputed)

# Scale the input data using the same scaler as used in training
input_data_scaled = scaler.transform(input_data_df)

# Make the prediction using the trained KNN model
if st.button("Predict"):
    try:
        prediction = knn_model.predict(input_data_scaled)  # Use the KNN model here
        st.write(f"The predicted class is: {'Malignant' if prediction[0] == 4 else 'Benign'}")

        if prediction[0] == 2:
            st.write("**2** is for probability of benign cancer")
        else:
            st.write("**4** is for probability of malignant cancer")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")

    
st.markdown(
    "<span style='color: blue;'><strong>Benign</strong></span>: Indicates that the abnormal cells found are not likely to lead to cancer.<br>"
    "<span style='color: red;'><strong>Malignant</strong></span>: Indicates that the cells are cancerous, and medical intervention is required to manage the "
    "condition, such as surgery, chemotherapy, radiation therapy, or other treatments.",
    unsafe_allow_html=True
)

#====================================================================================================
#Css Style
def set_background(png_file):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{png_file}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    /* Set text color to black for all text elements */
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: white;
        text-align: center; /* Center align text */
    }}
    .stButton > button {{
        background-color:#FBFBF6; /* Button color */
        color: white; /* Text color for button */
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stButton > button:hover {{
        background-color:#271CBD; /* Button hover color */
        color: white;
    }}
    .stSlider > div {{
        background-color: transparent;
    }}
    </style>
    """
