import re
import string

# Amharic stopwords – add more as needed
amharic_stopwords = set([
    'ነው', 'ነበር', 'እና', 'ወይም', 'እስከ', 'ለማን', 'በመሆኑ', 'እንደ', 'በተለይ',
    'እኔ', 'አንተ', 'እሱ', 'እሷ', 'እኛ', 'እናንተ', 'እነሱ'
])

# ✨ Remove emojis
def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# ✨ Remove URLs
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# ✨ Normalize Amharic punctuation, digits, case
def normalize_amharic(text):
    text = remove_emojis(text)
    text = remove_urls(text)
    text = re.sub(r'[0-9፩-፻]', '', text)  # Remove English/Amharic numbers
    amharic_punct = '፡።፣፤፥፦፧'
    translator = str.maketrans('', '', string.punctuation + amharic_punct)
    text = text.translate(translator)
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text

# ✨ Character-level normalization
def normalize_char_level_missmatch(token):
    text = token
    replacements = [
        ('[ሃኅኃሐሓኻ]', 'ሀ'), ('[ሑኁዅ]', 'ሁ'), ('[ኂሒኺ]', 'ሂ'), ('[ኌሔዄ]', 'ሄ'),
        ('[ሕኅ]', 'ህ'), ('[ኆሖኾ]', 'ሆ'), ('[ሠ]', 'ሰ'), ('[ሡ]', 'ሱ'), ('[ሢ]', 'ሲ'),
        ('[ሣ]', 'ሳ'), ('[ሤ]', 'ሴ'), ('[ሥ]', 'ስ'), ('[ሦ]', 'ሶ'), ('[ዓኣዐ]', 'አ'),
        ('[ዑ]', 'ኡ'), ('[ዒ]', 'ኢ'), ('[ዔ]', 'ኤ'), ('[ዕ]', 'እ'), ('[ዖ]', 'ኦ'),
        ('[ጸ]', 'ፀ'), ('[ጹ]', 'ፁ'), ('[ጺ]', 'ፂ'), ('[ጻ]', 'ፃ'), ('[ጼ]', 'ፄ'),
        ('[ጽ]', 'ፅ'), ('[ጾ]', 'ፆ'), ('(ሉ[ዋአ])', 'ሏ'), ('(ሙ[ዋአ])', 'ሟ'),
        ('(ቱ[ዋአ])', 'ቷ'), ('(ሩ[ዋአ])', 'ሯ'), ('(ሱ[ዋአ])', 'ሷ'), ('(ሹ[ዋአ])', 'ሿ'),
        ('(ቁ[ዋአ])', 'ቋ'), ('(ቡ[ዋአ])', 'ቧ'), ('(ቹ[ዋአ])', 'ቿ'), ('(ሁ[ዋአ])', 'ኋ'),
        ('(ኑ[ዋአ])', 'ኗ'), ('(ኙ[ዋአ])', 'ኟ'), ('(ኩ[ዋአ])', 'ኳ'), ('(ዙ[ዋአ])', 'ዟ'),
        ('(ጉ[ዋአ])', 'ጓ'), ('(ደ[ዋአ])', 'ዷ'), ('(ጡ[ዋአ])', 'ጧ'), ('(ጩ[ዋአ])', 'ጯ'),
        ('(ጹ[ዋአ])', 'ጿ'), ('(ፉ[ዋአ])', 'ፏ'), ('[ቊ]', 'ቁ'), ('[ኵ]', 'ኩ')
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    return text

# ✨ Tokenize by whitespace
def tokenize_amharic(text):
    return text.split()

# ✨ Remove stopwords
def remove_stopwords(tokens, stopwords=amharic_stopwords):
    return [word for word in tokens if word not in stopwords]

# ✨ Final clean function
def clean_amharic_text(text):
    text = normalize_amharic(text)
    tokens = tokenize_amharic(text)
    tokens = [normalize_char_level_missmatch(t) for t in tokens]
    tokens = remove_stopwords(tokens)
    return ' '.join(tokens)
