from transformers import pipeline
import streamlit as st

st.title("Sentiment Analysis")

model = pipeline('sentiment-analysis', model="finiteautomata/bertweet-base-sentiment-analysis")
sentence = st.text_input("Enter a sentence to analyze: ")
predicted = model(sentence)
st.write(f"Sentiment is: {predicted[0]['label']}.")

