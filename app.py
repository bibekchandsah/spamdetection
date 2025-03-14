# app.py
import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
import nltk

# Streamlit app
st.set_page_config(page_title="Spam Detection App", layout="wide", page_icon=":📩:")

# Download NLTK stopwords (run this only once)
nltk.download('stopwords')

# Load the saved model and TF-IDF vectorizer
model = joblib.load('spam_detector_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Stopword removal
    return text

# Streamlit app title
st.title("Spam Detection App")

# Input box for user to enter a message
user_input = st.text_area("Enter your message here:")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.error("Please enter a message.")
    else:
        # Preprocess the input text
        cleaned_message = preprocess_text(user_input)
        
        # Vectorize the input text
        vectorized_message = tfidf.transform([cleaned_message])
        
        # Predict using the loaded model
        prediction = model.predict(vectorized_message)
        
        # Display the result
        if prediction[0] == 1:
            st.error("This message is SPAM!")
        else:
            st.success("This message is HAM (not spam).")
            


with st.expander("ℹ️ About"):
    st.write("This is a spam detection app that uses a trained model to predict whether a message is spam or not.")
    st.write("The model was trained on a dataset of spam and ham messages.")
    st.write("The model was trained using a TF-IDF vectorizer and a logistic regression classifier.")

with st.expander("🚨Test Spam Messages"):
    st.write("Congratulations! You've won a free iPhone 15. Claim now by clicking the link: www.freeiphone.com ")
    st.write("URGENT: Your bank account has been compromised. Call 0800-123-4567 immediately to secure your funds.")
    st.write("Get a guaranteed loan approval within 24 hours. No credit check required. Reply YES to proceed.")
    st.write("You have been selected for a special discount on luxury watches. Visit www.luxurywatches.com to claim.")
    st.write("Exclusive offer: Win a trip to the Maldives for two. Text MALDIVES to 87654 to enter the contest.")
    st.write("Your mobile subscription is about to expire. Renew now to avoid service interruption. Call 0987-654-321.")
    st.write("Free entry into our weekly lottery. Just reply LOTTERY to 80080 to participate.")
    st.write("Someone in your area is interested in meeting you. Call 0700-123-456 to find out who!")
    st.write("Last chance to claim your £500 gift card. Click here: www.giftcardoffer.com ")
    st.write("You have an unread message from a secret admirer. Call 0870-567-890 to reveal their identity.")


with st.expander("💬Test Ham Messages"):
    st.write("Hey, how was your weekend? Did you get a chance to relax?")
    st.write("Don't forget to pick up milk on your way home.")
    st.write("I’ll be late for dinner tonight. Let me know if you want me to grab something for you.")
    st.write("Good morning! Hope you have a great day ahead.")
    st.write("Can you send me the notes from yesterday's class? I couldn’t make it.")
    st.write("Let’s meet at the coffee shop after work. I need to talk to you about something.")
    st.write("Thanks for the birthday wishes! I really appreciate it.")
    st.write("Are you still coming over this evening? Let me know so I can prepare.")
    st.write("Just wanted to check if you’re feeling better today. Take care!")
    st.write("Did you hear about the new movie release? We should watch it together.")


