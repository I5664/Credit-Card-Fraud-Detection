
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import pickle
# import lime
# import lime.lime_tabular
from sklearn.model_selection import train_test_split
import numpy as np
# import shap
# import base64
# from PIL import Image


st.title("Fraud Detection")

from streamlit.components.v1 import html
def nav_page(page_name, timeout_secs=3):
    nav_script = """
        <script type="text/javascript">
            function attempt_nav_page(page_name, start_time, timeout_secs) {
                var links = window.parent.document.getElementsByTagName("a");
                for (var i = 0; i < links.length; i++) {
                    if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
                        links[i].click();
                        return;
                    }
                }
                var elasped = new Date() - start_time;
                if (elasped < timeout_secs * 1000) {
                    setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
                } else {
                    alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
                }
            }
            window.addEventListener("load", function() {
                attempt_nav_page("%s", new Date(), %d);
            });
        </script>
    """ % (page_name, timeout_secs)
    html(nav_script)

col1, col4, col2, col5, col3 = st.columns([2,0.5,2,0.5,2])

TransactionMonth = col2.selectbox(
     'Enter Transaction Month',
     (1,2,3,4,5,6,7,8,9,10,11,12))

Amount = col3.number_input(
     'Enter Amount', value=29.87)

TypeOfTransaction_ = col2.selectbox(
     'Enter Transaction Type',
     ('Chip Transaction', 'Online Transaction', 'Swipe Transaction'))

if TypeOfTransaction_=='Swipe Transaction':
    TypeOfTransaction=0
elif TypeOfTransaction_=='Online Transaction':
    TypeOfTransaction=1
elif TypeOfTransaction_=='Chip Transaction':
    TypeOfTransaction=2

MerchantCity_ = col1.selectbox(
     'Enter Merchant City',
     ('Algiers',
     'Anaheim',
     'Brooklyn',
     'Chula Vista',
     'Claremont',
     'Corona',
     'La Verne',
     'Little Neck',
     'Ozone Park',
     'Pasadena',
     'Port au Prince',
     'Richmond',
     'Rome',
     'San Francisco',
     'Strasburg',
     'Upland'
     ))

if MerchantCity_=='Algiers':
    MerchantCity=4045
elif MerchantCity_=='Anaheim':
    MerchantCity=24
elif MerchantCity_=='Brooklyn':
    MerchantCity=3365
elif MerchantCity_=='Chula Vista':
    MerchantCity=200
elif MerchantCity_=='Claremont':
    MerchantCity=204
elif MerchantCity_=='Corona':
    MerchantCity=236
elif MerchantCity_=='La Verne':
    MerchantCity=551
elif MerchantCity_=='Little Neck':
    MerchantCity=600
elif MerchantCity_=='Ozone Park':
    MerchantCity=34
elif MerchantCity_=='Pasadena':
    MerchantCity=828
elif MerchantCity_=='Port au Prince':
    MerchantCity=885
elif MerchantCity_=='Richmond':
    MerchantCity=920
elif MerchantCity_=='Rome':
    MerchantCity=931
elif MerchantCity_=='San Francisco':
    MerchantCity=967
elif MerchantCity_=='Strasburg':
    MerchantCity=1079
elif MerchantCity_=='Upland':
    MerchantCity=1150

MerchantState_ = col1.selectbox(
     'Enter Merchant State',
     ('Algeria',
     'NY',
     'Haiti',
     'Italy',
     'OH',
     'CA'
     ))

if MerchantState_=='Algeria':
    MerchantState=68
elif MerchantState_=='NY':
    MerchantState=58
elif MerchantState_=='Haiti':
    MerchantState=25
elif MerchantState_=='Italy':
    MerchantState=34
elif MerchantState_=='OH':
    MerchantState=62
elif MerchantState_=='CA':
    MerchantState=14

HasChip_ = col3.selectbox(
     'Enter whether the card Has Chip',
     ('Yes','No'))

if HasChip_=='No':
    HasChip=0
elif HasChip_=='Yes':
    HasChip=1

CreditLimit = col2.number_input(
     'Enter Credit Limit',
     value=9700.0)

CurrentAge = col3.number_input(
     'Enter Current Age',
     value=71)

UserZip = col1.selectbox(
     'Enter User Zip',
     (7070, 10092, 91792, 9022, 24295, 94583))

# columns = ['Transaction Month', 'Amount', 'Type Of Transaction',
#        'Merchant City', 'Merchant State', 'Has Chip','Credit Limit', 'Current Age','User Zip']
# features = [[TransactionMonth, Amount, TypeOfTransaction, MerchantCity, MerchantState, HasChip, 
#             CreditLimit, CurrentAge,UserZip]]

columns = ['Merchant City', 'Transaction Month', 'Amount', 'Merchant State',
       'Use Chip', 'Has Chip', 'User Zip', 'Credit Limit', 'Retirement Age']
features = [[MerchantCity, TransactionMonth, Amount, MerchantState, TypeOfTransaction, HasChip, UserZip,
            CreditLimit, CurrentAge]]


with open('XGBoost.pkl', 'rb') as file:
    model = pickle.load(file)

df = pd.DataFrame(features, columns=columns)
print(df)

print(df.shape)
predictions = model.predict(df)
print(predictions)

df_ = pd.read_csv('Data_after_encode_1.csv')
df_ = df_.drop(['Unnamed: 0'], axis=1)


table_fraud = pd.DataFrame(
   data = [['Transaction Month', (TransactionMonth), 'Most fraudulent transactions occur in the second half of the year (i.e. between the months July and December)'],
           ['Transaction Type', TypeOfTransaction_, 'Most of the fraudulent transactions happen online'],
           ['Amount', Amount, 'On an average the transaction amount of fraudulent transactions is $130 and that of Non-fraudulent transactions is $54'],
           ['Merchant City', MerchantCity_, 'The offline fraudulent transactions frequently happened in the cities - Claremont and Corona'],
           ['Merchant State', MerchantState_, 'Most of the online and offline fraudulent transactions happened from the California State'],
           ['Has Chip', HasChip_, 'People holding with the chip found to make a fraudulent transaction'],
           ['User Zip', UserZip, 'Most of the fraudulent transactions are found to happen with the cards issued to the users from zipcodes - 91792, 91750 and 11363 (i.e) South California, LaVerne California and South East Newyork'],
           ['Current Age', CurrentAge, 'People in the age group 43 - 81 were found to involve in fraudulent transactions'],
           ['Credit Limit', CreditLimit, 'Most of the Fraudulent transactions happened in the credit cards with Credit limit ranging from $16000 - $132000'],
           ['prediction', predictions, 'asd']
           ],
   columns=('Features', 'User Inputs', 'Inferences from Analysis'))

table_fraud.index = np.arange(1, len(table_fraud) + 1)

table_fraud.to_csv('table_fraud.csv', index=False)


if st.button("Predict", type='primary'):
    nav_page("Result")