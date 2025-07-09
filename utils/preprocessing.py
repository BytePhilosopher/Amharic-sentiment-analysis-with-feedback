import re
import string

def clean_amharic(text):
    text = str(text)
    text = re.sub(r'[{}]'.format(string.punctuation), '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
