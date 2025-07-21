import re
import string

# Sample Amharic stopwords – expand as needed
amharic_stopwords = set([
    'ነው', 'ነበር', 'እና', 'ወይም', 'እስከ', 'ለማን', 'በመሆኑ', 'እንደ', 'በተለይ',
    'እኔ', 'አንተ', 'እሱ', 'እሷ', 'እኛ', 'እናንተ', 'እነሱ'
])

def normalize_amharic(text):
    """
    Normalize Amharic text by:
    - Removing punctuation
    - Lowercasing
    - Removing numbers
    """
    # Remove English and Arabic numerals
    text = re.sub(r'[0-9፩-፻]', '', text)
    
    # Remove English and Amharic punctuation
    amharic_punct = '፡።፣፤፥፦፧'
    translator = str.maketrans('', '', string.punctuation + amharic_punct)
    text = text.translate(translator)

    # Normalize spacing and lowercase
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)

    return text

def tokenize_amharic(text):
    """
    Tokenize Amharic text using whitespace.
    """
    return text.split()

def remove_stopwords(tokens, stopwords=amharic_stopwords):
    """
    Remove predefined Amharic stopwords.
    """
    return [word for word in tokens if word not in stopwords]

def clean_amharic_text(text):
    """
    Full pipeline: normalize → tokenize → remove stopwords → return cleaned string
    """
    text = normalize_amharic(text)
    tokens = tokenize_amharic(text)
    tokens = remove_stopwords(tokens)
    return ' '.join(tokens)
