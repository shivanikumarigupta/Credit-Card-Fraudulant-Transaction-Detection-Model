## We will make web app

import numpy as np
import pandas as pd  ## data cleaning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

## Load data

credited_card_df = pd.read_csv('creditcard.csv')
  # credited_card_df.head()

## seperate legitimate and fraudlent transactions
  # 1st will be
legit = credited_card_df[credited_card_df.Class == 0]
  # 2nd
fraud = credited_card_df[credited_card_df['Class']==1]

## Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
credited_card_df = pd.concat([legit_sample,fraud], axis=0)

## split data into training and testing sets
X = credited_card_df.drop('Class',axis=1)   ## Independented feature
Y = credited_card_df['Class']               ## dependented feature
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=2)

## train logistic regression model
model = LogisticRegression()
model.fit(X_train,Y_train)

## evaluate model performance
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test),Y_test)

## Web app
st.title("Credit Card Fraud Detection Model")
input_df = st.text_input('Enter All Required Features Values')
input_df_splited = input_df.split(',')

submit = st.button("Submit")

if submit:
    features = np.asarray(input_df_splited, dtype =np.float64)
    prediction = model.predict(features.reshape(1,-1))

    if prediction[0] ==0:
        st.write("Legitimate Transaction")

    else:
        st.write("Fradulent Transaction")
