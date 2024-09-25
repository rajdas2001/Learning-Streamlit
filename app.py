import pandas as pd
import streamlit as st
import numpy as np

st.title("Exploratory Data Analysis")
chart_data = pd.DataFrame(
    np.random.rand(20,3),columns=['a','b','c']
)
st.line_chart(chart_data)