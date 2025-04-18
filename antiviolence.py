import streamlit as st
import tensorflow as tf
import numpy as np
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

#Labels
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
hate_labels = ['Hate Speech', 'Offensive', 'Neither']
violence_labels = ['sexual violence', 'physical violence', 'eet']

#Loading the models on streamlit
@st.cache_resource
def load_models():
    #Loading with the hugging face set up
    emotion_model = TFAutoModelForSequenceClassification.from_pretrained(
        "models/emotion_roberta_model", local_files_only=True
    )
    hate_model = TFAutoModelForSequenceClassification.from_pretrained(
        "models/hate_roberta_model_hf", local_files_only=True
    )
    violence_model = TFAutoModelForSequenceClassification.from_pretrained(
        "models/violence_roberta_model_hf", local_files_only=True
    )

    # Loading tokenizers
    tokenizer_emotion = AutoTokenizer.from_pretrained("models/emotion_tokenizer", local_files_only=True)
    tokenizer_hate = AutoTokenizer.from_pretrained("models/hate_tokenizer", local_files_only=True)
    tokenizer_violence = AutoTokenizer.from_pretrained("models/violence_tokenizer", local_files_only=True)

    return emotion_model, hate_model, violence_model, tokenizer_emotion, tokenizer_hate, tokenizer_violence

#Prediction function
def predict_label(text, model, tokenizer, labels):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(inputs, training=False)
    pred = tf.argmax(outputs.logits, axis=1).numpy()[0]
    return labels[pred]

#Loading the models to predict user input
emotion_model, hate_model, violence_model, tokenizer_emotion, tokenizer_hate, tokenizer_violence = load_models()

#User Interface
st.title("🧠 Multi-Stage Text Classifier (Emotion → Hate/Violence)")
st.markdown("This app first predicts the **main emotion** of the text. If the emotion is **anger** or **fear**, it checks for hate or violence.")

user_input = st.text_area("✍️ Enter a user comment or post:", height=150)

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            results = {}
            emotion = predict_label(user_input, emotion_model, tokenizer_emotion, emotion_labels)
            results["Emotion"] = emotion

            if emotion == "anger":
                hate = predict_label(user_input, hate_model, tokenizer_hate, hate_labels)
                results["Hate"] = hate

            if emotion == "fear":
                violence = predict_label(user_input, violence_model, tokenizer_violence, violence_labels)
                results["Violence"] = violence

        #Results
        st.subheader("🔎 Classification Results")
        for key, value in results.items():
            st.markdown(f"**{key}:** {value}")
