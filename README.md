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

### 🔸 Algorithms Used:
- **Multinomial Naive Bayes**
- **Gaussian Naive Bayes**
- **Logistic Regression**

All models were trained using both `CountVectorizer` and `TF-IDF` features.

### 🔸 Evaluation Metrics:
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**  
- **Confusion Matrix**

> ✅ Evaluation done using `classification_report` and `confusion_matrix` from scikit-learn.  
> ✅ Logistic Regression with TF-IDF gave the best performance.

---

## 🧠 Best Model
- **Model**: Logistic Regression (TF-IDF)
- **Saved As**: `data/best_model.pkl`

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
│ ├── X_tfidf.npy
│ ├── best_model.pkl
│ └── cleaned_imdb_dataset.csv
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
- Model training with Multinomial NB, Gaussian NB, and Logistic Regression  
- Evaluated all models on key metrics  
- Identified and saved best-performing model (`Logistic Regression`)  
- Updated GitHub repo with daily progress

---

## 🔮 Upcoming Tasks
- Visualize model performance using Seaborn/Matplotlib  
- Deploy model using Streamlit (Web App)  
- Prepare demo video or presentation  
- Add model prediction interface  
- Write detailed blog/documentation  

---

## 👨‍💻 Author
**Ajaz Ahmed**  
[GitHub Profile](https://github.com/AjazAhmed293)