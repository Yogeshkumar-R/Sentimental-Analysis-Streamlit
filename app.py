import streamlit as st
import pandas as pd
from transformers import pipeline

# Load the pre-trained sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis")

# Streamlit app layout
st.title('Sentiment Analysis App')

# User input
user_text = st.text_area('Enter text for sentiment analysis:', height=200)

# Perform sentiment analysis when the user clicks the button
if st.button('Analyze Sentiment'):
    # Predict sentiment using the loaded model
    prediction = sentiment_analysis(user_text)[0]
    sentiment = prediction['label']
    
    # Display the result
    st.write(f'Sentiment: {sentiment}')

# Additional feature: Display some sample data
st.subheader('Sample Data for Sentiment Analysis')
sample_data = {
    'Text': ['I love this product!', 'This movie was terrible. Waste of time.'],
    'Sentiment': ['Positive', 'Negative']
}
df = pd.DataFrame(sample_data)
st.write(df)
