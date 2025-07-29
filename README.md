# 📊 Sentiment Analysis Using NLP

## 📌 Objective
To build a machine learning model that classifies movie reviews from the IMDB dataset as **Positive** or **Negative** using Natural Language Processing techniques.

---

## 🧩 Dataset
- **Name**: IMDB Dataset of 50K Movie Reviews
- **Source**: [Kaggle - IMDB Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size**: 50,000 labeled reviews (balanced classes)

---

## ⚙️ Preprocessing Steps
Text cleaning and preprocessing was performed to prepare the data for modeling. Steps include:

1. Converted all reviews to lowercase  
2. Removed digits, punctuation, and HTML tags  
3. Removed extra whitespaces  
4. Tokenized text using `word_tokenize` from NLTK  
5. Removed English stopwords  
6. Created a new column `clean_review` containing the cleaned text

---

## 🛠 Technologies Used
- Python
- Pandas
- NLTK
- Regular Expressions (`re`)
- Git + GitHub

---

## 📁 Folder Structure

sentiment-analysis-nlp/
├── data/
│ ├── IMDB Dataset.csv
│ └── IMDB_cleaned.csv (optional - if saved after cleaning)
├── project.py
├── problem_staement.txt
├── README.md

---

## 🔮 Upcoming Tasks
- Feature Extraction using CountVectorizer and TF-IDF
- Model training (Logistic Regression, Naive Bayes, etc.)
- Evaluation (Accuracy, Precision, Recall)
- Deployment using Streamlit or Flask

---

## 👨‍💻 Author
Ajaz Ahmed  
[GitHub Profile](https://github.com/AjazAhmed293)