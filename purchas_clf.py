import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier, XGBRegressor, XGBRFRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pickle

st.set_page_config(layout='wide',page_title='classificagtion')

st.title('Alvin purchase classification')

train = pd.read_csv('Train.csv')
data = st.file_uploader('Upload data set')

def create_binary_cols(content):
  if content == False:
    content = 0
  elif content == True:
    content = 1
  elif content == 'N':
    content = 0
  elif content == 'Y':
    content = 1
  elif content == 'Male':
    content = 0
  elif content == 'Female':
    content = 1
  return content	


selected_option = st.selectbox("Select an option", ["Random Forest", "XGBClassifier"])
if selected_option == "Random Forest":
            model_load_path = "model.pkl"
            with open(model_load_path, 'rb') as file:
                model = pickle.load(file)

            if data is not None:
                    test = pd.read_csv(data)
                    train["USER_GENDER"] = train["USER_GENDER"].apply(lambda x: "Male" if pd.isna(x) else x)
                    test["USER_GENDER"] = test["USER_GENDER"].apply(lambda x: "Male" if pd.isna(x) else x)
                    # Impute the missing age entries with the median of that column
                    train_median_value = np.median(train['USER_AGE'].dropna())
                    train['USER_AGE'] = train['USER_AGE'].fillna(train_median_value)

                    test_median_value = np.median(test['USER_AGE'].dropna())
                    test['USER_AGE'] = test['USER_AGE'].fillna(test_median_value)

                    train["train"] = 1
                    test["train"] = 0

                    all_data = pd.concat([train, test])

                    all_data = pd.get_dummies(all_data, prefix_sep="_", columns=['MERCHANT_NAME'])

                    train = all_data[all_data["train"] == 1]
                    test = all_data[all_data["train"] == 0]

                    train = train.drop(['MERCHANT_CATEGORIZED_AT','PURCHASED_AT','USER_ID', 'Transaction_ID', "train"], axis=1)
                    test = test.drop(['MERCHANT_CATEGORIZED_AT','PURCHASED_AT','USER_ID', "train", "MERCHANT_CATEGORIZED_AS"], axis=1)

                    train['USER_GENDER'] = train['USER_GENDER'].apply(create_binary_cols)
                    test['USER_GENDER'] = test['USER_GENDER'].apply(create_binary_cols)

                    # Is_purchase_paid_via_mpesa_send_money column convert:
                    train['IS_PURCHASE_PAID_VIA_MPESA_SEND_MONEY'] = train['IS_PURCHASE_PAID_VIA_MPESA_SEND_MONEY'].apply(create_binary_cols)
                    test['IS_PURCHASE_PAID_VIA_MPESA_SEND_MONEY'] = test['IS_PURCHASE_PAID_VIA_MPESA_SEND_MONEY'].apply(create_binary_cols)

                    X = train.drop(["MERCHANT_CATEGORIZED_AS"], axis=1)
                    y = train["MERCHANT_CATEGORIZED_AS"]
            if st.button('Predict Category'):
                    # Get the predicted result for the test Data
                predictions = model.predict(test.drop("Transaction_ID", axis=1))
                test["predictions"] = predictions
                sub = test[["Transaction_ID",  "predictions"]]
                sub = pd.get_dummies(sub, columns=['predictions'])
                # # remove the p
                sub.columns = sub.columns.str.replace('predictions_','')
                st.write('Result')
                st.write(test[["Transaction_ID",  "predictions"]])