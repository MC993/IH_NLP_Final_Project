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
        print("I'm glad to hear you are having a great time. May you feel this way for a long time.")
    elif max_mood == 'anger':
        print('I understand how you can feel frustrated and I wish you feel better soon.')
    elif max_mood == 'fear':
        print('I understand how you can be concerned. Please connect with a loved one to seek support.')
    elif max_mood == 'love':
        print('Feeling love is the best feeling one could experience. May this long last for you and your loved one.')
    elif max_mood == 'surprise':
        print('You seem surprised! I hope the surprise was a good one for you!')
    else:
        print("I'm sorry to hear you are sad. If you need support, please reach out to one of your loved ones.")
    return moods