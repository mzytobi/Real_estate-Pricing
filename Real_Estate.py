import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import time
import base64

# Set the theme and layout
st.set_page_config(
    page_title="Real Estate House Pricing Prediction App",
    page_icon="🏠",
    layout="wide",  # Use 'centered' or 'wide'
    initial_sidebar_state="expanded"  # Use 'expanded' or 'collapsed'
)

# Custom CSS
css = """
body {
    background-color: #f0f0f0;
    font-family: "Helvetica Neue", Arial, sans-serif;
}
h1 {
    color: #007BFF;
}
"""
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Load the pre-trained model and preprocessing objects
model = pickle.load(open('LGBMC_model.pkl', 'rb'))
scaler = pickle.load(open('scal.pkl', 'rb'))
encoder = pickle.load(open('enc.pkl', 'rb'))

# Define the 'prep' function for data preparation
def prep(new_data):
    # Make a copy of the input DataFrame to avoid modifying the original data
    new_data_copy = new_data.copy()

    # Dropping the 'ID' column and storing it in 'cust'
    cust = new_data_copy['ID']
    new_data_copy.drop(['ID'], axis=1, inplace=True)

    # Encoding object variables for ML
    cat_cols = ['title', 'loc']
    new_data_copy[cat_cols] = encoder.transform(new_data_copy[cat_cols])

    # Scaling the numeric features
    col_names = new_data_copy.columns
    new_data_copy = scaler.transform(new_data_copy)
    new_data_copy = pd.DataFrame(new_data_copy, columns=col_names)

    return cust, new_data_copy

def main():
      ## side bar
    img=Image.open("ETm.jpg")
    st.sidebar.image(img.resize((1280,780)))
    
    ##Real estate name
    st.sidebar.title("")
   
     #Web name
    st.title("Real Estate House Price Prediction WebApp")
    
     # Add an image
    image_path = "thp.jpg"# Update with the correct path
    banner_width = 1200  # Set the width in pixels
    banner_height = 400  # Set the height in pixels 
    st.image(image_path, width= banner_width)

    

    # Upload a CSV file and get user input
    uploaded_file = st.sidebar.file_uploader("Upload A CSV", type='csv')

    if uploaded_file is not None:
        # Read the uploaded CSV file into a dataframe
        df = pd.read_csv(uploaded_file)

        # Perform data preparation using the 'prep' function
        cust, prepared_data = prep(df)

        # Display the uploaded data
        st.subheader("Uploaded Data:")
        st.write(df)

        # Display the prepared data
        st.subheader("Prepared Data:")
        st.write(prepared_data)
        
        # Add Filter Options
        st.sidebar.subheader("Filter Options")
        price_range_1M_to_2_5M = st.sidebar.checkbox("1M to 2.5M")
        price_range_3M_to_5M = st.sidebar.checkbox("3M to 5M")


        # Perform model predictions
        if st.button('Predict'):
            with st.spinner("Predicting..."):
                pred = model.predict(prepared_data)
                results = pd.DataFrame({'ID': cust, "Price_Prediction": pred})
                time.sleep(2)
                st.subheader("Price Predictions:")
                st.write(results)
               # Display download button without clearing the displayed dataframe
            st.markdown(get_download_link(results), unsafe_allow_html=True)
                 # Filter IDs based on selected price ranges
            if price_range_1M_to_2_5M:
                filtered_ids_1M_to_2_5M = results.loc[(results['Price_Prediction'] >= 1000000) & (results['Price_Prediction'] <= 2500000), 'ID'].tolist()
                st.subheader("IDs with Price between 1M and 2.5M:")
                st.write(filtered_ids_1M_to_2_5M)

            if price_range_3M_to_5M:
                filtered_ids_3M_to_5M = results.loc[(results['Price_Prediction'] >= 3000000) & (results['Price_Prediction'] <= 5000000), 'ID'].tolist()
                st.subheader("IDs with Price between 3M and 5M:")
                st.write(filtered_ids_3M_to_5M)

            # Provide download buttons for prediction results
                csv1 = results.to_csv(index=False)
                st.download_button('Download Predictions', csv1, file_name='Price_predictions.csv')
                 # Show "Task completed successfully!" after the prediction is completed
            st.success("Price Prediction completed successfully!")
def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predicted_new_values.csv">Download Predicted Values</a>'
    return href

if __name__ == "__main__":
    main()
