import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re 
import nltk
import string
nltk.download('punkt')
nltk.download('stopwords')
dataset=pd.read_csv(r"C:\Users\ajaz6\OneDrive\Desktop\sentiment_analysis_nlp\data\IMDB Dataset.csv")
print(dataset.head())
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
def save_cleaned_data(dataset, file_path):
    dataset.to_csv(file_path, index=False)  # Save the cleaned dataset to a CSV file
save_cleaned_data(dataset, r"C:\Users\ajaz6\OneDrive\Desktop\sentiment_analysis_nlp\data\cleaned_imdb_dataset.csv")  # Specify the file path to save the cleaned data
print("Data cleaning complete and saved to cleaned_imdb_dataset.csv")  # Confirmation message
dataset = pd.read_csv(r"C:\Users\ajaz6\OneDrive\Desktop\sentiment_analysis_nlp\data\cleaned_imdb_dataset.csv")
print(dataset.head())
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=5000, ngram_range=(1, 2))  # Initialize CountVectorizer with max features and n-grams
X_count= cv.fit_transform(dataset['cleaned_text']).toarray()  # Fit and transform the cleaned text data
print(X_count.shape)  # Print the shape of the resulting array (50000,5000)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1,2),max_features=5000)  # Initialize TfidfTransformer
X_tfidf = tfidf.fit_transform(dataset['cleaned_text']).toarray()  # Fit and transform the cleaned text data
print(X_tfidf.shape)  # Print the shape of the resulting array (50000,5000)
import numpy as np
np.save('data/X_count.npy', X_count)
np.save('data/X_tfidf.npy', X_tfidf)