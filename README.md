# 📊 Sentiment Analysis Using NLP

## 📌 Objective
To build a machine learning model that classifies movie reviews from the IMDB dataset as **Positive** or **Negative** using Natural Language Processing techniques.

---

## 🧩 Dataset
- **Name**: IMDB Dataset of 50K Movie Reviews
- **Source**: [Kaggle - IMDB Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size**: 50,000 labeled reviews (balanced classes)
- **Location**: `/data/IMDB Dataset.csv`

---

## ⚙️ Preprocessing Steps
Text cleaning and preprocessing was performed to prepare the data for modeling. Steps include:

1. Converted all reviews to lowercase  
2. Removed digits, punctuation, and HTML tags  
3. Removed extra whitespaces  
4. Tokenized text using `word_tokenize` from NLTK  
5. Removed English stopwords  
6. Created a new column `clean_review` containing the cleaned text  
7. Cleaned dataset saved as `data/IMDB_cleaned.csv`

---

## 🧮 Feature Engineering

Text data was transformed into numerical vectors using:

### 🔹 CountVectorizer
- Converts text into a bag-of-words representation
- Config: `max_features=5000`, `ngram_range=(1, 2)`
- Output saved as: `data/X_count.npy`

### 🔹 TfidfVectorizer
- Converts text into TF-IDF-weighted vectors
- Config: `max_features=5000`, `ngram_range=(1, 2)`
- Output saved as: `data/X_tfidf.npy`

---

## 🛠 Technologies Used
- Python
- Pandas
- NLTK
- Scikit-learn
- Regular Expressions (`re`)
- Git + GitHub

---

## 📁 Folder Structure

sentiment-analysis-nlp/
├── data/
│ ├── IMDB Dataset.csv
│ ├── IMDB_cleaned.csv
│ ├── X_count.npy
│ └── X_tfidf.npy
├── project.py
├── problem_staement.txt
├── README.md

---

## 🔮 Upcoming Tasks
- Train/test split
- Logistic Regression and Naive Bayes modeling
- Model evaluation (accuracy, precision, recall, confusion matrix)
- Deployment using Streamlit or Flask

---

## 👨‍💻 Author
Ajaz Ahmed  
[GitHub Profile](https://github.com/AjazAhmed293)
