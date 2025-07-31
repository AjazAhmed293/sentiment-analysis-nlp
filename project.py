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
# print(X_tfidf.shape)  # Print the shape of the resulting array (50000,5000)
import numpy as np
np.save('data/X_count.npy', X_count)
np.save('data/X_tfidf.npy', X_tfidf)
dataset=pd.read_csv(r"C:\Users\ajaz6\OneDrive\Desktop\sentiment_analysis_nlp\data\IMDB Dataset.csv")
X_count = np.load('data/X_count.npy')  # Load the count vectorized data
X_tfidf = np.load('data/X_tfidf.npy')  # Load the TF-IDF vectorized data
print(dataset['sentiment'].unique())  # Get unique sentiment labels
Y=dataset['sentiment'].map({'positive': 1, 'negative': 0})  # Map sentiment labels to binary values
print(Y)  # Print the transformed sentiment labels
from sklearn.model_selection import train_test_split
X_train_count, X_test_count, Y_train, Y_test = train_test_split(X_count, Y, test_size=0.2, random_state=42)  # Split the count vectorized data into training and testing sets
X_train_tfidf, X_test_tfidf, Y_train, Y_test = train_test_split(X_tfidf, Y, test_size=0.2, random_state=42)  # Split the TF-IDF vectorized data into training and testing sets
print(X_train_count.shape, X_test_count.shape, Y_train.shape, Y_test.shape)  # Print the shapes of the training and testing sets
from sklearn.naive_bayes import MultinomialNB
# Initialize the Multinomial Naive Bayes classifier
nb_count = MultinomialNB()  # For count vectorized data
nb_tfidf = MultinomialNB()  # For TF-IDF vectorized data
# Fit the model on the training data
nb_count.fit(X_train_count, Y_train)  # Fit the model using count vectorized data
nb_tfidf.fit(X_train_tfidf, Y_train)  # Fit the model using TF-IDF vectorized data
# Predict on the test data
Y_pred_count = nb_count.predict(X_test_count)  # Predict using count vectorized data
Y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)  # Predict using TF-IDF vectorized data
# Evaluate the model performance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Calculate accuracy for count vectorized data
accuracy_count = accuracy_score(Y_test, Y_pred_count)  # Accuracy for count vectorized data
print(f'Count Vectorizer Accuracy: {accuracy_count:.4f}')  # Print accuracy for count vectorized data
# Calculate accuracy for TF-IDF vectorized data
accuracy_tfidf = accuracy_score(Y_test, Y_pred_tfidf)  # Accuracy for TF-IDF vectorized data
print(f'TF-IDF Vectorizer Accuracy: {accuracy_tfidf:.4f}')
# Print classification report for count vectorized data
print("Classification Report for Count Vectorizer:")  # Print classification report for count vectorized data
print(classification_report(Y_test, Y_pred_count))  # Classification report for count vectorized
# Print classification report for TF-IDF vectorized data
print("Classification Report for TF-IDF Vectorizer:")  # Print classification report for TF-IDF vectorized data
print(classification_report(Y_test, Y_pred_tfidf))  # Classification report for TF-IDF vectorized data
# Print confusion matrix for count vectorized data  
print("Confusion Matrix for Count Vectorizer:")  # Print confusion matrix for count vectorized data
print(confusion_matrix(Y_test, Y_pred_count))  # Confusion matrix for count vectorized data
# Print confusion matrix for TF-IDF vectorized data
print("Confusion Matrix for TF-IDF Vectorizer:")  # Print confusion matrix for TF-IDF vectorized data
print(confusion_matrix(Y_test, Y_pred_tfidf))  # Confusion matrix for TF-IDF vectorized data