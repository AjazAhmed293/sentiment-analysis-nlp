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
Text cleaning and preprocessing were performed using NLTK and regex. Steps include:
1. Converted all reviews to lowercase  
2. Removed digits, punctuation, and HTML tags  
3. Removed extra whitespaces  
4. Tokenized text using `word_tokenize`  
5. Removed English stopwords  
6. Combined cleaned tokens back to strings  
7. Cleaned text stored in-memory and used for feature extraction

---

## 🧮 Feature Engineering

### 🔹 CountVectorizer
- Converts text into a bag-of-words representation  
- Config: `max_features=5000`, `ngram_range=(1, 2)`  
- Output saved as: `data/X_count.npy`

### 🔹 TfidfVectorizer
- Converts text into TF-IDF-weighted vectors  
- Config: `max_features=5000`, `ngram_range=(1, 2)`  
- Output saved as: `data/X_tfidf.npy`

---

## 🧪 Model Training & Evaluation

### 🔸 Algorithm: Multinomial Naive Bayes
- Trained using both `X_count` and `X_tfidf` feature sets  
- Evaluated on 80/20 train-test split

### 🔸 Evaluation Metrics:
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**  
- **Confusion Matrix**

> ✅ All metrics printed and compared using `classification_report` and `confusion_matrix` from scikit-learn.

---

## 🛠 Technologies Used
- Python
- Pandas
- NLTK
- NumPy
- Scikit-learn
- Regular Expressions (`re`)
- Git + GitHub

---

## 📁 Folder Structure

sentiment-analysis-nlp/
├── data/
│ ├── IMDB Dataset.csv
│ ├── X_count.npy
│ └── X_tfidf.npy
├── project.py
├── problem_staement.txt
├── README.md
└── .gitignore

---

## ✅ Completed Tasks
- Problem statement defined  
- Dataset loaded and explored  
- Text preprocessing and cleaning  
- Feature engineering using CountVectorizer & TF-IDF  
- Saved transformed features as `.npy`  
- Model training with Naive Bayes  
- Model evaluated with key metrics  
- GitHub repo updated daily

---

## 🔮 Upcoming Tasks
- Logistic Regression model  
- Model performance comparison  
- Experiment with Tfidf-based training  
- Visualize performance using Matplotlib or Seaborn  
- Prepare for deployment using Streamlit

---

## 👨‍💻 Author
Ajaz Ahmed  
[GitHub Profile](https://github.com/AjazAhmed293)