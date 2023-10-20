import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib

#load the model from disk
model = joblib.load(r"random_forest_classifier_e.joblib")


def main():
    #Setting Application title
    st.title('Ecom Customer Churn-Prediction App')

      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer churn in the given Ecom use case.
    The application is functional for both online prediction and batch data prediction.
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('ecom_app.jpg')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")
        #Based on our optimal features selection

        st.subheader("Demographic data")
        Gender = st.selectbox('Gender :', ('Female', 'Male'))
        MaritalStatus = st.selectbox('MaritalStatus:', ('Single', 'Married', 'Divorced'))
        
        st.subheader("Activity data")
        Tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=32, value=0)
        PreferedOrderCat = st.selectbox('PreferedOrderCat:', ('Laptop & Accessory', 'Mobile', 'Others', 'Fashion', 'Grocery'))
        HourSpendOnApp = st.number_input('The amount of hours spent on the app', min_value=0, max_value=5, value=0)
        NumberOfDeviceRegistered = st.number_input('The amount of devices registered by customer', min_value=1, max_value=6, value=1)
        PreferredLoginDevice = st.selectbox('PreferredLoginDevice:', ('Phone', 'Computer'))

        st.subheader("Geographic data")
        CityTier = st.selectbox('CityTier:', (3, 1, 2))
        NumberOfAddress = st.number_input('NumberOfAddresses registered by customer', min_value=1, max_value=11, value=1)
        WarehouseToHome = st.number_input('Distance of customer from Warehouse', min_value=5, max_value=25, value=5)
       

        st.subheader("Payment data")
        PreferredPaymentMode = st.selectbox('PreferredPaymentMode',('Debit Card', 'UPI', 'CC', 'COD', 'E wallet'))
        OrderAmountHikeFromlastYear = st.number_input('The hike rate in amount of order by the customer',min_value=10, max_value=25, value=10)
        CashbackAmount = st.number_input('CashbackAmount', min_value=100, max_value=300, value=150)
        DaySinceLastOrder = st.number_input('Days passed since last ordered from App:', min_value=1, max_value=30, value=5)
        OrderCount = st.number_input('Total orders made from App so far:', min_value=1, max_value=15, value=2)
        CouponUsed =  st.number_input('CouponsUsed', min_value=0, max_value=10, value=2)
    
        st.subheader("Feedback data")
        SatisfactionScore = st.number_input('How satisfied is the cust with the app.', min_value=1, max_value=5, value=3)
        Complain = st.selectbox('Registered any complaint.', (0, 1))
        
        data = {
                'HourSpendOnApp': HourSpendOnApp,
                'NumberOfDeviceRegistered': NumberOfDeviceRegistered,
                'SatisfactionScore':SatisfactionScore,
                'NumberOfAddress': NumberOfAddress,
                'OrderAmountHikeFromlastYear': OrderAmountHikeFromlastYear,
                'CouponUsed': CouponUsed,
                'OrderCount': OrderCount,
                'DaySinceLastOrder': DaySinceLastOrder,
                'CashbackAmount': CashbackAmount,
                'Tenure': Tenure,
                'WarehouseToHome': WarehouseToHome,
                'PreferredLoginDevice': PreferredLoginDevice,
                'CityTier': CityTier,
                'PreferredPaymentMode':PreferredPaymentMode,
                'Gender': Gender,
                'PreferedOrderCat': PreferedOrderCat,
                'MaritalStatus':MaritalStatus,
                'Complain':Complain
                }
        

        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)

        # st.write(features_df.dtypes)

        # Create input_data for the model 
        input_df_columns = ['HourSpendOnApp', 'NumberOfDeviceRegistered', 'SatisfactionScore',
                                'NumberOfAddress', 'OrderAmountHikeFromlastYear', 'CouponUsed',
                                'OrderCount', 'DaySinceLastOrder', 'CashbackAmount', 
                                'Tenure','WarehouseToHome', 'PreferredLoginDevice_Phone', 
                                'CityTier_2','CityTier_3', 'PreferredPaymentMode_COD',
                                'PreferredPaymentMode_Debit Card', 'PreferredPaymentMode_E wallet',
                                'PreferredPaymentMode_UPI', 'Gender_1', 'PreferedOrderCat_Grocery',
                                'PreferedOrderCat_Laptop & Accessory', 'PreferedOrderCat_Mobile', 'PreferedOrderCat_Others',
                                'MaritalStatus_1', 'MaritalStatus_2','Complain_1']
        
        num_inps = [data['HourSpendOnApp'], data['NumberOfDeviceRegistered'], data['SatisfactionScore'],
                    data['NumberOfAddress'], data['OrderAmountHikeFromlastYear'], data['CouponUsed'],
                    data['OrderCount'], data['DaySinceLastOrder'], data['CashbackAmount'],
                    data['Tenure'], data['WarehouseToHome']]
        
        if data['PreferredLoginDevice']=='Computer':
             device=[0]
        else:
             device=[1]

        if data['CityTier']== 1:
             city = [0,0]
        elif data['CityTier']==2:
             city = [1,0] 
        else:
             city = [0,1]

        if data['PreferredPaymentMode'] == 'CC':
            payment = [0, 0, 0, 0]
        elif data['PreferredPaymentMode'] == 'COD':
            payment = [1, 0, 0, 0]
        elif data['PreferredPaymentMode'] == 'Debit Card':
            payment = [0, 1, 0, 0]
        elif data['PreferredPaymentMode'] == 'E Wallet':
            payment = [0, 0, 1, 0]
        else:
            payment = [0, 0, 0, 1]

        if data['Gender'] == 'Female':
            gender = [0]
        else:
            gender = [1]

        if data['PreferedOrderCat'] == 'Fashion':
            ord_cat = [0,0,0,0]
        elif data['PreferedOrderCat'] == 'Grocery':
            ord_cat = [1,0,0,0]
        elif data['PreferedOrderCat'] == 'Laptop & Accessory':
            ord_cat = [0,1,0,0]
        if data['PreferedOrderCat'] == 'Mobile':
            ord_cat = [0,0,1,0]
        else:
            ord_cat = [0,0,0,1]

        if data['MaritalStatus'] == 'Single':
            marital = [0,0]
        elif data['MaritalStatus'] == 'Married':
            marital = [1,0]
        else:
            marital = [0,1]

        if data['Complain'] == 0:
            complain = [0]
        else:
            complain = [1]

        final_input = num_inps + device + city + payment + gender + ord_cat + marital + complain 
             
        # st.write(final_input)
             

        input_df = pd.DataFrame(data= [final_input],
                                columns = input_df_columns )

        # Predict if the customer churns
        prediction = model.predict(input_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the customer will churn from the Ecom app service.')
            else:
                st.success('No, the customer is happy with Ecom App Services.')


    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        st.write('Uploaded file can be similarly processed and batch prediction can be undertaken.')
        pass

if __name__ == '__main__':
        main()