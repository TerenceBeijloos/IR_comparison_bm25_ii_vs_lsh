import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure necessary NLTK data is downloaded
nltk.download('punkt_tab')
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

class Lexer:
    def __init__(self, remove_stop_words=True, apply_lemmatization=True, additional_stopwords=None):
        self.apply_lemmatization = apply_lemmatization
        self.lemmatizer = WordNetLemmatizer() if apply_lemmatization else None
        self.remove_stop_words = remove_stop_words
        self.stop_words = set(stopwords.words('english')) if remove_stop_words else set()
        if additional_stopwords:
            self.stop_words.update(additional_stopwords)

    # Tokenize string
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)  # Remove non-word characters, keeping spaces

        # Tokenize
        tokens = word_tokenize(text)

        # Process tokens
        processed_tokens = []
        
        for token in tokens:
            # Skip stop words if enabled
            if self.remove_stop_words and token in self.stop_words:
                continue
            
            # Lemmatize token if enabled
            if self.apply_lemmatization:
                token = self.lemmatizer.lemmatize(token)
                
            processed_tokens.append(token)

        return processed_tokens