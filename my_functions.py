#NLP text cleaner

import re

def clean_text(text):

    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\brt\b', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


#stopwords removal

import re
import nltk
from nltk.corpus import stopwords 

def stopword_remove(text):
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    return text


#emotion decoder

def mooder(text):
    """
    This function will return the mood of the text
        returns_tensors="tf" will return the tokens as a TensorFlow tensor
        max_length=512 will limit the number of tokens to 512
        truncation=True will truncate the text if it exceeds the max_length
        padding="longest" will pad the text to the longest sequence in the batch
    """
    encoded_text = tokenizer(text, return_tensors="tf", max_length=512, truncation=True, padding="longest")
    output = model(encoded_text)
    scores = output.logits[0].numpy()
    moods = {
        "anger": scores[0],
        "joy": scores[1],
        "fear": scores[2],
        "love": scores[3],
        "surprise": scores[4],
        "sadness": scores[5]
    }

    max_mood = max(moods, key=moods.get)
    print(f"Your mood appear to be {max_mood}")

    if max_mood == 'joy':
        print(f"The detected emotion is {max_mood}. No further investigation required.")
    elif max_mood == 'anger':
        print(f"The detected emotion is {max_mood}. Further investigation required.")
    elif max_mood == 'fear':
        print(f"The detected emotion is {max_mood}. Further investigation may be required.")
    elif max_mood == 'love':
        print(f"The detected emotion is {max_mood}. No further investigation required.")
    elif max_mood == 'surprise':
        print(f"The detected emotion is {max_mood}. No further investigation required.")
    else:
        print(f"The detected emotion is {max_mood}. No further investigation may be required.")
    return moods