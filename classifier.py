from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

import streamlit as st

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

model = RandomForestClassifier(n_estimators=1000)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
final_acc = acc*100
st.title("Iris Flower Classification using Random Forest")
st.write("Made by Raj Das.")
st.text("")
st.write(f"Accuracy of the model is {final_acc}%.")

st.subheader("Try it out now!")

st.subheader("Input Features")
sepal_length = st.text_input("Enter sepal length in cm: ")
sepal_width = st.text_input("Enter sepal width in cm: ")
petal_length = st.text_input("Enter petal length in cm: ")
petal_width = st.text_input("Enter petal width in cm: ")


st.subheader("Output")
if sepal_length and sepal_width and petal_length and petal_width:
    input_data = [[float(sepal_length),float(sepal_width),float(petal_length),float(petal_width)]]
    val = model.predict(input_data)
    if val == 0:
        st.write("Iris-setosa")
    elif val == 1:
        st.write("Iris-versicolor")
    else:
        st.write("Iris-virginica")

