import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re 
import nltk
import string
# nltk.download('punkt')
# nltk.download('stopwords')
dataset=pd.read_csv(r"C:\Users\ajaz6\OneDrive\Desktop\sentiment_analysis_nlp\data\IMDB Dataset.csv")
# print(dataset.head())
def clean_text(text):
    text=text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    tokens=word_tokenize(text)  # Tokenize the text
    stop_words = set(stopwords.words('english'))  # Get English stop words
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    return' '.join(tokens)  # Join tokens back into a string
dataset['cleaned_text'] = dataset['review'].apply(clean_text)  # Apply cleaning function to the 'review' column
print(dataset.head())