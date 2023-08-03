import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
    st.title("Real Estate House Pricing Prediction App")

    # Upload a CSV file and get user input
    uploaded_file = st.file_uploader("Upload a CSV", type='csv')

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

        # Perform model predictions
        if st.button('Predict'):
            pred = model.predict(prepared_data)
            results = pd.DataFrame({'ID': cust, "Churn_Prediction": pred})
            st.subheader("Churn Predictions:")
            st.write(results)

            # Provide download buttons for prediction results
            csv1 = results.to_csv(index=False)
            st.download_button('Download Predictions', csv1, file_name='churn_predictions.csv')

if __name__ == "__main__":
    main()
