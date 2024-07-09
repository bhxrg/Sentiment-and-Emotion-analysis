import streamlit as st
from transformers import pipeline

# Load the sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

# Load the emotion analysis model
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Define a simple preprocessing function
def preprocess_tweet(text):
    return text.lower()

# Function to classify emotions and sentiments
def classify_tweet(tweet):
    preprocessed_tweet = preprocess_tweet(tweet)
    
    # Perform sentiment analysis
    sentiment_analysis = sentiment_pipeline(preprocessed_tweet)[0]
    sentiment_label = sentiment_analysis['label']
    sentiment_score = sentiment_analysis['score']
    
    # Perform emotion analysis
    emotion_results = emotion_pipeline(preprocessed_tweet)[0]
    
    # Extract emotions and their scores
    emotions = [result['label'] for result in emotion_results]
    emotion_scores = [result['score'] for result in emotion_results]
    
    return {
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "emotions": emotions,
        "emotion_scores": emotion_scores
    }

# Define main function for Streamlit app
def main():
    st.set_page_config(page_title="Sentiment and Emotion Analysis", page_icon=":bar_chart:", layout="wide")

    # Custom CSS for background and text colors
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #f5f5f5;  /* Main background color */
            color: #333333;  /* Main text color */
        }
        .sidebar .sidebar-content {
            background-color: #d8e2dc;  /* Sidebar background color */
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;  /* Input field background color */
            color: #333333;  /* Input field text color */
        }
        .stButton>button {
            background-color: #4CAF50;  /* Button background color */
            color: white;  /* Button text color */
        }
        .stProgress>div>div>div>div {
            background-color: #4CAF50;  /* Progress bar color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Sentiment and Emotion Analysis")

    # Text input for entering a sample sentence or tweet text
    sentence = st.text_input("Enter a tweet text:", "")

    # Button to trigger analysis
    if st.button("Analyze"):
        if sentence:
            # Perform classification
            classification_result = classify_tweet(sentence)

            # Display the input text
            st.subheader("Input Text:")
            st.write(sentence)

            # Display sentiment analysis
            st.subheader("Sentiment Analysis:")
            sentiment_label = classification_result['sentiment_label']
            sentiment_score = classification_result['sentiment_score']
            st.write(f"Sentiment: {sentiment_label} ({sentiment_score:.2f})")

            # Display sentiment score breakdown side by side
            col1, col2, col3 = st.columns(3)
            with col1:
                if sentiment_label == "POSITIVE":
                    st.write(f"Positive: {sentiment_score*100:.2f}%")
            with col2:
                if sentiment_label == "POSITIVE":
                    st.write(f"Neutral: {(1 - sentiment_score)*50:.2f}%")
            with col3:
                if sentiment_label == "POSITIVE":
                    st.write(f"Negative: {(1 - sentiment_score)*50:.2f}%")
            
            with col1:
                if sentiment_label == "NEGATIVE":
                    st.write(f"Positive: {(1 - sentiment_score)*50:.2f}%")
            with col2:
                if sentiment_label == "NEGATIVE":
                    st.write(f"Neutral: {(1 - sentiment_score)*50:.2f}%")
            with col3:
                if sentiment_label == "NEGATIVE":
                    st.write(f"Negative: {sentiment_score*100:.2f}%")
            
            if sentiment_label not in ["POSITIVE", "NEGATIVE"]:
                with col2:
                    st.write(f"Neutral: {sentiment_score*100:.2f}%")

            # Display emotion analysis
            st.subheader("Emotion Analysis:")
            for emotion, score in zip(classification_result['emotions'], classification_result['emotion_scores']):
                st.write(f"{emotion.capitalize()}: {score:.2f}")
                st.progress(score)

if __name__ == '__main__':
    main()